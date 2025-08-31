import os
import cv2
import copy
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from pipeline_sd15 import StableDiffusionPipeline
from unet_sd15_ip_adapter import Dift_UNet2DConditionModel
from ip_adapter.ip_adapter import IPAdapter
from transformers import CLIPTokenizer
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import inversion_pipeline_sd15_ipadapter
import traceback

def main():
    # --- Configuration ---
    base_model_path = "/root/autodl-tmp/stable-diffusion-v1-5/"
    # base_model_path = "/root/autodl-tmp/Realistic_Vision_V4.0_noVAE/"
    image_encoder_path = "/root/autodl-tmp/IP-Adapter/models/image_encoder/"
    ip_ckpt = "/root/autodl-tmp/IP-Adapter/models/ip-adapter_sd15.bin"
    tsv_path = "/root/autodl-tmp/mip-adapter_finegrain/assets/prompt.tsv"
    image_dir = "/root/autodl-tmp/test_dataset"
    mask_dir = "/root/autodl-tmp/test_dataset_mask"
    output_dir = "/root/autodl-tmp/mip-adapter_finegrain/output/batch_test_sd15_paste_latent_42"
    os.makedirs(output_dir, exist_ok=True)

    # --- Hyperparameters ---
    resolution = 512
    num_inference_steps = 50
    scale = 0.6
    num_samples = 1
    feature_merged_time = None
    merged_layers = list(range(9, 17))
    merged_times = list(range(20, 51))
    _lambda = 0.5
    seed_generation = 42

    # --- Load prompts ---
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    # --- Initialize models ---
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    unet = Dift_UNet2DConditionModel.from_pretrained(
        base_model_path, subfolder="unet", torch_dtype=torch.float16
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        unet=unet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_slicing()
    pipe.to("cuda")
    pipe.scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False
    )

    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt=ip_ckpt, device="cuda")

    # --- Batch processing ---
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        idx = row["index"] if "index" in row else str(_)
        prompt = row["prompt"]
        ref_prompt = row.get("subject", prompt)
        try:
            # Load reference image and mask
            ref_img_path = os.path.join(image_dir, f"{idx}.jpg")
            mask_img_path = os.path.join(mask_dir, f"{idx}_mask.jpg")
            inv_img_np = np.array(load_image(ref_img_path).resize((resolution, resolution)))
            image = Image.open(ref_img_path).resize((resolution, resolution))
            mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            mask = torch.from_numpy(mask).float().unsqueeze(0).to("cuda")

            # Get embeddings and perform inversion
            prompt_embeds, neg_embeds = ip_model.get_sd15_embeddings(
                pil_image=image,
                prompt=ref_prompt,
                negative_prompt="",
                scale=scale,
                num_samples=num_samples
            )
            ref_latents = inversion_pipeline_sd15_ipadapter.ddim_inversion_v1_5_ipadapter(
                pipe=pipe,
                x0=inv_img_np,
                prompt_embeds=prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                negative_prompt_embeds=neg_embeds
            )

            # Snapshot reference dift features
            ref_dift = {
                k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                for k, v in pipe.unet.dift_latent_store.dift_features.items()
            }
            tgt_dift = pipe.unet.dift_latent_store.dift_features

            # Configure IP-Adapter with dift merging
            ip_model.set_dift_ipadapter(
                ref_dift=ref_dift,
                tgt_dift=tgt_dift,
                merged_layers=merged_layers,
                merged_times=merged_times,
                ref_mask=mask
            )

            # Generate images
            generated = ip_model.generate_with_dift(
                pil_image=image,
                num_samples=num_samples,
                num_inference_steps=num_inference_steps,
                seed=seed_generation,
                prompt=prompt,
                ref_prompt=ref_prompt,
                scale=scale,
                ref_dift=ref_dift,
                feature_merged_time=feature_merged_time,
                _lambda=_lambda,
                ref_latents=ref_latents,
                mask=mask,
                target_dift=tgt_dift,
                use_ini_latents=True,
            )

            # Save outputs
            for j, out_img in enumerate(generated):
                out_path = os.path.join(output_dir, f"{idx}_{j}.png")
                out_img.save(out_path)

        except Exception as e:
            print(f"[Error] idx={idx} processing failed: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
