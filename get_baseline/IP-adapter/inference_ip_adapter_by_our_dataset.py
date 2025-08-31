import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterXL

tsv_path = "/root/autodl-tmp/prompt.tsv"
image_folder = "/root/autodl-tmp/test_dataset"
output_folder = "/root/autodl-tmp/ip_adapter_our_dataset_42"

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/root/autodl-tmp/IP-Adapter/sdxl_models/image_encoder"
ip_ckpt = "/root/autodl-tmp/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

num_samples = 1
num_inference_steps = 50
seed = 42
scale = 0.6

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

df = pd.read_csv(tsv_path, sep="\t")

os.makedirs(output_folder, exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df), desc="progress"):
    try:
        index = row["index"]
        prompt = row["prompt"]

        image_path = os.path.join(image_folder, f"{index}.jpg")
        if not os.path.exists(image_path):
            print(f"warning: image not found: {image_path}, skip")
            continue

        image = Image.open(image_path).convert("RGB").resize((512, 512))

        images = ip_model.generate(
            pil_image=image,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            prompt=f"best quality, high quality, {prompt}",
            scale=scale
        )

        for i, img in enumerate(images):
            filename = f"{index}_{i}.png"
            save_path = os.path.join(output_folder, filename)
            img.save(save_path)
            print(f"save image to {save_path}")

    except Exception as e:
        print(f"error when processing index={row.get('index')}: {e}")
        continue
