
import argparse
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from models.clip_model import CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.modeling_utils import ModelMixin

from textual_inversion_dataset import TextualInversionDataset
from utils import *
import json

from accelerate import Accelerator

import timm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------x

check_min_version("0.15.0.dev0")

logger = get_logger(__name__)

#########################
def set_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU: {gpu_id}")

class AdapterLayer(nn.Module):
    def __init__(self, embed_dim, fused_dim, beta=0.3):
        super(AdapterLayer, self).__init__()
        self.beta = beta  # Adapter의 영향력을 조절하는 상수
        self.gamma = nn.Parameter(torch.zeros(1))  # 학습 가능한 스칼라 파라미터
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        self.fc1 = nn.Linear(fused_dim, embed_dim)

    def forward(self, y, fused_embedding):

        batch_size, seq_length, embed_dim = y.size()  # [batch_size, seq_length, embed_dim]
        fused_embedding_projected = self.fc1(fused_embedding)  # [num_fused_tokens, embed_dim]
        fused_embedding_expanded = fused_embedding_projected.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_fused_tokens, embed_dim]
        combined_input = torch.cat([y, fused_embedding_expanded], dim=1)  # [batch_size, seq_length + num_fused_tokens, embed_dim]

        S_output, _ = self.self_attention(
            query=combined_input,
            key=combined_input,
            value=combined_input,
        )

        S_output = S_output[:, :seq_length, :]  # [batch_size, seq_length, embed_dim]

        y = y + self.beta * torch.tanh(self.gamma) * S_output
        return y
    
    
class UNetWithAdapter(ModelMixin):
    def __init__(self, unet, adapter_layer):
        super(UNetWithAdapter, self).__init__()
        self.unet = unet
        self.adapter_layer = adapter_layer
        self.config = unet.config

        # Freeze all parameters in UNet
        for param in self.unet.parameters():
            param.requires_grad = False

        # Enable gradients only for cross-attention key and value
        for name, param in self.unet.named_parameters():
            if "attn2.to_k" in name or "attn2.to_v" in name:  # Cross-attention key and value
                param.requires_grad = True

    def forward(self, sample, timestep, encoder_hidden_states, fused_embedding=None, **kwargs):
        # Ensure adapter gets the input
        if self.adapter_layer is not None:
            if fused_embedding is None:
                raise ValueError("fused_embedding must be provided for the AdapterLayer.")
            # Modify encoder_hidden_states using the adapter
            encoder_hidden_states = self.adapter_layer(encoder_hidden_states, fused_embedding)

        # Pass through UNet
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )

        
    def save_config(self, save_directory):

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        if isinstance(self.config, dict):
            config_to_save = self.config
        else:
            config_to_save = dict(self.config)  # FrozenDict를 dict로 변환

        # config.json 파일로 저장
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_to_save, f, indent=2)

        # AdapterLayer-specific configuration 저장
        adapter_config = {
            "beta": self.adapter_layer.beta,
            "gamma": self.adapter_layer.gamma.item(),
        }
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)




def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, step, placeholder_tokens):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    validation_output_dir=os.path.join(args.output_dir,"validation")
    if not os.path.exists(validation_output_dir):
        os.makedirs(validation_output_dir)

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet_with_adapter),
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    for i, prompt in enumerate(args.validation_prompt):
        prompt=prompt.format(' '.join(placeholder_tokens))

        image = pipeline(prompt, num_inference_steps=50, generator=generator,num_images_per_prompt=args.num_validation_images).images
        image=image_grid(image,1,len(image))
        image.save(os.path.join(validation_output_dir,f'{"_".join(prompt.split(" "))}_step_{step}.jpg'))
    
    del pipeline
    torch.cuda.empty_cache()


def save_progress(text_encoder, placeholder_tokens, placeholder_token_ids, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds_dict = dict()
    token_embeds=accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for token, id in zip(placeholder_tokens, placeholder_token_ids):
        learned_embeds = token_embeds[id]
        learned_embeds_dict[token] = learned_embeds.detach().cpu()
    torch.save(learned_embeds_dict, save_path)
    with open(os.path.join(os.path.dirname(save_path), "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
    
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.query_projection = nn.Linear(embed_dim, embed_dim)

        self.key_value_projection = nn.Linear(embed_dim, embed_dim)

        self.linear_projection = nn.Linear(embed_dim, 2 * embed_dim)  

    def forward(self, arcface_embedding, initialize_embeds):
        device = arcface_embedding.device
        initialize_embeds = initialize_embeds.to(device)

        arcface_embedding = arcface_embedding.unsqueeze(0) 
        query = self.query_projection(arcface_embedding)    

        key_value = self.key_value_projection(initialize_embeds)  

        attention_output, _ = self.cross_attention(
            query,                    
            key_value.unsqueeze(0),    
            key_value.unsqueeze(0)  
        ) 

        linear_output = self.linear_projection(attention_output.squeeze(0))  


        fused_embedding = linear_output.view(2, -1)

        return fused_embedding

def train():

    set_gpu(0)

    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # ArcFace 모델 로드 및 임베딩 추출
    print("Before ArcFace embedding extraction")
    arcface_model = timm.create_model("hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3", pretrained=True).eval()
    image_path = "/home/prml/Jin/Main/examples/input_images/00006/00006.jpg"  # 이미지 경로
    image = Image.open(image_path).convert("RGB")
    image = image.resize((112, 112))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    arcface_embedding = F.normalize(arcface_model(image_tensor), dim=1)

    print("ArcFace embedding extracted:", arcface_embedding.shape)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        setup_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer 로드
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Scheduler와 모델 로드
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet",revision=args.revision)
    adapter = AdapterLayer(embed_dim=1024, fused_dim=2048, beta=0.4) # 적절한 embed_dim으로 AdapterLayer 초기화

    # UNet에 AdapterLayer를 통합한 UNetWithAdapter 생성
    unet_with_adapter = UNetWithAdapter(unet, adapter)

    placeholder_tokens = []
    placeholder_token_ids = []
    for id in range(args.n_persudo_tokens):
        new_token=args.placeholder_token+f'_v{id}'
        num_added_tokens = tokenizer.add_tokens(new_token)
        if num_added_tokens ==0:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        placeholder_tokens.append(new_token)
        placeholder_token_ids.append(tokenizer.convert_tokens_to_ids(new_token))

    text_encoder.resize_token_embeddings(len(tokenizer))

    if args.initialize_tokens is not None:
        assert len(args.initialize_tokens)==args.n_persudo_tokens,"The number of `initialize_tokens` is not equal to `n_persudo_tokens`"
        initialize_embeds=token_cross_init(args.initialize_tokens,tokenizer,text_encoder)
    else:
        initialize_embeds=celeb_names_cross_init(args.celeb_path,tokenizer,text_encoder,args.n_persudo_tokens)
    
    text_encoder.get_input_embeddings().weight.data[placeholder_token_ids]=initialize_embeds
    initialize_embeds=initialize_embeds.clone().detach().to(accelerator.device)


    arcface_dim_expander = torch.nn.Linear(512, 1024).to(arcface_embedding.device)
    arcface_embedding_expand = arcface_dim_expander(arcface_embedding)  # 레이어 호출


    arcface_embedding_resized = arcface_embedding_expand.expand(2, -1)
    

    print(f"Resized ArcFace Embedding Shape: {arcface_embedding_resized.shape}")  
    
    initialize_embeds = initialize_embeds.to(arcface_embedding_resized.device)
    
    cross_attention_fusion = CrossAttentionFusion(embed_dim=1024, num_heads=8).to(accelerator.device)
    arcface_embedding_resized = arcface_embedding_resized.to(accelerator.device)
    initialize_embeds = initialize_embeds.to(accelerator.device)
    fused_embedding = cross_attention_fusion(arcface_embedding_resized, initialize_embeds)

    print("Fused Embedding Shape:", fused_embedding.shape)  



    vae.requires_grad_(False)

    unet_with_adapter.adapter_layer.requires_grad_(True)

    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)


    if args.gradient_checkpointing:
        unet_with_adapter.train()
        text_encoder.gradient_checkpointing_enable()
        unet_with_adapter.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17."
                )
            unet_with_adapter.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimizer 초기화
    optimizer = torch.optim.AdamW(
        list(text_encoder.get_input_embeddings().parameters()) + 
        list(unet_with_adapter.adapter_layer.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset 및 DataLoader 생성
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_tokens=placeholder_tokens,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_with_adapter.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                
                # `fused_embedding`를 생성하는 코드
                fused_embedding = cross_attention_fusion(arcface_embedding_resized, initialize_embeds)

                # `unet_with_adapter`의 forward 호출
                model_pred = unet_with_adapter(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    fused_embedding=fused_embedding
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                token_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
                # 공분산 행렬 계산 (PyTorch로 수행)
                covariance_matrix = torch.cov(initialize_embeds.T)

                # 공분산 행렬 정규화
                zeta = 2e-5
                covariance_matrix += torch.eye(covariance_matrix.size(0), device=covariance_matrix.device) * zeta

                # 공분산 행렬의 역행렬 계산
                inv_covariance_matrix = torch.linalg.pinv(covariance_matrix)

                func = token_embeds[placeholder_token_ids] - initialize_embeds  # [2, 1024]
                distance_squared_raw = func @ inv_covariance_matrix @ func.T

                # 거리 계산 (수치적 안정성 추가)
                distance_squared = torch.clamp(torch.diagonal(distance_squared_raw), min=1e-8)  # 대각선만 사용
                mahalanobis_distance = torch.sqrt(distance_squared)

                # Regularization Loss 계산
                reg_loss = mahalanobis_distance.mean() * args.reg_weight

                loss=F.mse_loss(model_pred,target,reduction='mean')+reg_loss

                accelerator.backward(loss, retain_graph=True)


                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                vocab = torch.arange(len(tokenizer))
                index_no_updates = (vocab != placeholder_token_ids[0])
                for token_id in placeholder_token_ids[1:]:
                    index_no_updates = torch.logical_and(index_no_updates, vocab != token_id)
                with torch.no_grad():
                    token_embeds[index_no_updates] = orig_embeds_params[index_no_updates]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_tokens, placeholder_token_ids, accelerator, args, save_path)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "embed_loss": reg_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet_with_adapter,
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(args.output_dir)
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(text_encoder, placeholder_tokens, placeholder_token_ids, accelerator, args, save_path)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initialize_tokens", 
        type=str, 
        default=None, 
        required=False, 
        nargs="*",
        help="Tokens to use as initializer words."
    )

    parser.add_argument(
        "--celeb_path", 
        type=str, 
        default=None, 
        required=False, 
        help="Celeb basis file that contains celeb names."
    )
    
    parser.add_argument(
        "--n_persudo_tokens",
        type=int,
        default=2,
        required=True,
        help="Number of persudo tokens to use in training.",
    )
    
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=1e-4,
        required=False,
        help="Weight of the regularization term.",
    )

    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-07, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_prompt_file",
        type=str,
        default=None,
        help="A file containing several prompts that are used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.validation_prompt is not None and args.validation_prompt_file is not None:
        raise ValueError("`--validation_prompt` cannot be used with `--validation_prompt_file`")
    
    if args.validation_prompt is not None:
        args.validation_prompt=[args.validation_prompt]

    if args.validation_prompt_file is not None:
        with open(args.validation_prompt_file,'r') as f:
            args.validation_prompt=f.read().splitlines()
    
    if args.initialize_tokens is not None and args.celeb_path is not None:
        raise ValueError("`--initialize_tokens` cannot be used with `--celeb_path`")
    
    if args.initialize_tokens is None and args.celeb_path is None:
        raise ValueError("`--initialize_tokens` and `--celeb_path` cannot both be empty.")
    
    return args

if __name__ == "__main__":
    train()