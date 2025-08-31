import os
import sys
import datetime
import torch
import pandas as pd
from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from utils.utils import load_image, load_mask
from pipelines.pipeline_stable_diffusion_freecustom import StableDiffusionFreeCustomPipeline
from freecustom.mrsa import MultiReferenceSelfAttention
from freecustom.hack_attention import hack_self_attention_to_mrsa

def override_cfg(cfg, ref_image_path, target_prompt, subject):
    cfg.ref_image_infos       = {ref_image_path: "a " + subject}
    cfg.target_prompt         = target_prompt
    cfg.use_null_ref_prompts  = True
    return cfg

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    # --- 1) load base config ---
    cfg = OmegaConf.load("configs/config_stable_diffusion.yaml")

    # --- 2) read your prompt.tsv ---
    df = pd.read_csv(
        "/root/autodl-tmp/FreeCustom/prompt.tsv",
        sep="\t", usecols=["index", "prompt", "subject"],
        dtype={"index": str, "prompt": str, "subject": str}
    )

    # --- 3) prepare results dir ---
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "test_batch_42_fix")
    os.makedirs(results_dir, exist_ok=True)

    # --- 4) set up device & model once ---
    torch.cuda.set_device(cfg.gpu)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False
    )
    model = StableDiffusionFreeCustomPipeline.from_pretrained(
        cfg.model_path, scheduler=scheduler,
        cache_dir="/root/autodl-tmp/sd_model"
    ).to(device)
    model.safety_checker = None

    IMAGE_DIR = "/root/autodl-tmp/test_dataset"
    MASK_DIR  = "/root/autodl-tmp/test_dataset_mask"

    # --- 5) loop with tqdm and file-existence check ---
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
        try:
            idx    = row["index"]
            prompt = row["prompt"]
            subject = row["subject"]
            
            img_path  = os.path.join(IMAGE_DIR, f"{idx}.jpg")
            mask_path = os.path.join(MASK_DIR,  f"{idx}_mask.jpg")  
    
            if not os.path.exists(img_path):
                print(f"[Warning] image not found: {img_path}, skipping.")
                continue
            if not os.path.exists(mask_path):
                print(f"[Warning] mask  not found: {mask_path}, skipping.")
                continue
    
            cfg = override_cfg(cfg, img_path, prompt, subject)
    
            ref_img = load_image(img_path, device)
            ref_msk = load_mask(mask_path, device)
            ref_lat = model.image2latent(ref_img)
    
            prompts          = [cfg.target_prompt, "a " + subject]
            negative_prompts = [cfg.negative_prompt, cfg.negative_prompt]
            viz_cfg = OmegaConf.load("configs/config_for_visualization.yaml")
            viz_cfg.results_dir = results_dir
            viz_cfg.ref_image_infos = cfg.ref_image_infos
            OmegaConf.save(cfg, os.path.join(results_dir, "config.yaml"))
    
            for seed in cfg.seeds:
                seed_everything(seed)
    
                mrsa = MultiReferenceSelfAttention(
                            start_step     = cfg.start_step,
                            end_step       = cfg.end_step,
                            layer_idx      = cfg.layer_idx,
                            ref_masks      = [ref_msk],
                            mask_weights   = cfg.mask_weights,
                            style_fidelity = cfg.style_fidelity,
                            viz_cfg        = viz_cfg
                      )
                hack_self_attention_to_mrsa(model, mrsa)
    
                noise   = torch.randn_like(ref_lat)
                latents = torch.cat([noise, ref_lat], dim=0)
    
                out = model(
                    prompt          = prompts,
                    latents         = latents,
                    guidance_scale  = getattr(cfg, "guidance_scale", 7.5),
                    negative_prompt = negative_prompts,
                ).images[0]
    
                out_path = os.path.join(results_dir, f"{idx}_0.png")
                out.save(out_path)
    
        except Exception as e:
            print(f"[Error] Failed to process index {row['index']}: {e}")
            continue

