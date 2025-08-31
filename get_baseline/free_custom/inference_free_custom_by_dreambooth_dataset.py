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
    cfg.ref_image_infos = {ref_image_path: "a " + subject}
    print(cfg.ref_image_infos)
    cfg.target_prompt = target_prompt
    cfg.use_null_ref_prompts = False
    return cfg

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    # --- 1) load base config ---
    cfg = OmegaConf.load("configs/config_stable_diffusion.yaml")

    # --- 2) load dreambooth prompts ---
    df = pd.read_csv("/root/autodl-tmp/FreeCustom/dreambooth_prompts.tsv", sep="\t")

    # --- 3) prepare output base dir ---
    output_base = "/root/autodl-tmp/free_custom_dreambooth_42_fix"
    os.makedirs(output_base, exist_ok=True)

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

    # --- 5) process by name ---
    grouped = df.groupby("name")

    for name, group in grouped:
        # locate images
        image_dir = os.path.join("/root/autodl-tmp/dreambooth_single_image/dataset", name)
        images = [f for f in os.listdir(image_dir) if not f.endswith("_mask.jpg") and f.lower().endswith(('.jpg', '.png'))]
        masks = [f for f in os.listdir(image_dir) if f.endswith("_mask.jpg")]

        if not images or not masks:
            print(f"[Warning] Missing image or mask for {name}, skipping.")
            continue

        ref_img_path = os.path.join(image_dir, images[0])
        ref_msk_path = os.path.join(image_dir, masks[0])

        # load ref image and mask
        ref_img = load_image(ref_img_path, device)
        ref_msk = load_mask(ref_msk_path, device)
        ref_lat = model.image2latent(ref_img)

        # prepare output folder
        output_dir = os.path.join(output_base, name)
        os.makedirs(output_dir, exist_ok=True)

        # visualization config
        viz_cfg = OmegaConf.load("configs/config_for_visualization.yaml")
        viz_cfg.results_dir = output_dir
        viz_cfg.ref_image_infos = {ref_img_path: ""}

        # process each prompt for this name
        for i, row in tqdm(group.iterrows(), total=len(group), desc=f"Generating for {name}"):
            subject = row["subject"]
            prompt = row["prompt"]
            prompt_prefix = prompt[:40].replace(" ", "_").replace("/", "_")

            cfg = override_cfg(cfg, ref_img_path, prompt, subject)

            prompts = [cfg.target_prompt, "a " + subject]
            negative_prompts = [cfg.negative_prompt, cfg.negative_prompt]

            for seed in cfg.seeds:
                seed_everything(seed)

                mrsa = MultiReferenceSelfAttention(
                    start_step=cfg.start_step,
                    end_step=cfg.end_step,
                    layer_idx=cfg.layer_idx,
                    ref_masks=[ref_msk],
                    mask_weights=cfg.mask_weights,
                    style_fidelity=cfg.style_fidelity,
                    viz_cfg=viz_cfg
                )
                hack_self_attention_to_mrsa(model, mrsa)

                noise = torch.randn_like(ref_lat)
                latents = torch.cat([noise, ref_lat], dim=0)

                out = model(
                    prompt=prompts,
                    latents=latents,
                    guidance_scale=getattr(cfg, "guidance_scale", 7.5),
                    negative_prompt=negative_prompts,
                ).images[0]

                output_filename = f"{i+1:03d}_{subject.replace(' ', '_')}_{prompt_prefix}.png"
                output_path = os.path.join(output_dir, output_filename)
                out.save(output_path)
