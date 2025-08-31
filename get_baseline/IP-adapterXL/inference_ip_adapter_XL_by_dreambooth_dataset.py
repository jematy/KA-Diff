import os
import re
import torch
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterPlusXL
from pathlib import Path
from tqdm import tqdm

tsv_path = "/root/autodl-tmp/IP-adapter/dreambooth_prompts.tsv"
dataset_root = "/root/autodl-tmp/dreambooth_single_image/dataset"
output_root = "/root/autodl-tmp/ip_adapterXL_dreambooth_42"

base_model_path = "SG161222/RealVisXL_V1.0"
image_encoder_path = "/root/autodl-tmp/IP-Adapter/models/image_encoder"
ip_ckpt = "/root/autodl-tmp/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
device = "cuda"

num_samples = 1
num_inference_steps = 50
seed = 42
scale = 0.5

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    use_safetensors=True, variant="fp16",
).to(device)

ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

df = pd.read_csv(tsv_path, sep="\t")

os.makedirs(output_root, exist_ok=True)

grouped = df.groupby("name")

for name, group in tqdm(grouped, desc="Total progress"):
    try:
        folder_path = os.path.join(dataset_root, name)
        output_dir = os.path.join(output_root, name)
        os.makedirs(output_dir, exist_ok=True)

        ref_images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith((".jpg", ".png")) and not f.endswith("_mask.jpg")]
        if not ref_images:
            print(f"warning: no valid image found for {name}, skip")
            continue

        ref_image_path = os.path.join(folder_path, ref_images[0])
        image = Image.open(ref_image_path).convert("RGB").resize((512, 512))

        for i, row in enumerate(group.itertuples(), start=1):
            subject, prompt = row.subject, row.prompt
            full_prompt = f"best quality, high quality, {prompt}"
            prompt_prefix = re.sub(r'\W+', '_', prompt[:40]).strip("_")

            images = ip_model.generate(
                pil_image=image,
                num_samples=num_samples,
                num_inference_steps=num_inference_steps,
                seed=seed,
                prompt=full_prompt,
                scale=scale
            )

            for j, img in enumerate(images):
                filename = f"{i:03d}_{subject.replace(' ', '_')}_{prompt_prefix}.png"
                save_path = os.path.join(output_dir, filename)
                img.save(save_path)
                print(f"save image to {save_path}")

    except Exception as e:
        print(f"error when processing {name}: {e}")
        continue