# ID-EA (SMC 2025)

# Abstract
Recently, personalized portrait generation with a text-to-image diffusion model has significantly advanced with Textual Inversion, emerging as a promising approach for creat- ing high-fidelity personalized images. Despite its potential, cur- rent Textual Inversion methods struggle to maintain consistent facial identity due to semantic misalignments between textual and visual embedding spaces regarding identity. We introduce ID-EA, a novel framework that guides text embeddings to align with visual identity embeddings, thereby improving identity preservation in a personalized generation. ID-EA comprises two key components: the ID-driven Enhancer (ID-Enhancer) and the ID-conditioned Adapter (ID-Adapter). First, the ID- Enhancer integrates identity embeddings with a textual ID anchor, refining visual identity embeddings derived from a face recognition model using representative text embeddings. Then, the ID-Adapter leverages the identity-enhanced embedding to adapt the text condition, ensuring identity preservation by adjusting the cross-attention module in the pre-trained UNet model. This process encourages the text features to find the most related visual clues across the foreground snippets. Extensive quantitative and qualitative evaluations demonstrate that ID-EA substantially outperforms state-of-the-art methods in identity preservation metrics while achieving remarkable computational efficiency, generating personalized portraits ap- proximately 15 times faster than existing approaches.


# Setup

Our code mainly bases on [Diffusers-Textual Inversion](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) and relies on the [diffusers](https://github.com/huggingface/diffusers) library.

To set up the environment, please run:

```bash
conda create -n ci python=3.10
conda activate ci

pip install -r requirements.txt


```Training
python train_arcross_adapter_kv.py     --save_steps 100      --only_save_embeds     --placeholder_token "<00011>"     --train_batch_size 8    --scale_lr    --n_persudo_tokens 2    --reg_weight "8e-4"   --learning_rate 0.0015     --max_train_step 320   --train_data_dir "./examples/input_images/00011"    --celeb_path "./examples/wiki_names_v2.txt"     --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base"     --output_dir "./logs/00011/learned_embeddings"

```Inference
python test_cross_init.py  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --num_inference_steps 50 --learned_embedding_path "/home/prml/Jin/Main/logs/00011_test/learned_embeddings/learned_embeds.bin"  --prompt "{} wearing an oversized sweater" --save_dir "./logs/00011_test/images"  --num_images_per_prompt 32  --n_iter 1


