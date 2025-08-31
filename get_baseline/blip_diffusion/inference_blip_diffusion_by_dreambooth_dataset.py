import os
import pandas as pd
from diffusers.pipelines.blip_diffusion.pipeline_blip_diffusion import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch
from PIL import Image

blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "Salesforce/blipdiffusion", torch_dtype=torch.float16, ignore_mismatched_sizes=True
).to("cuda")
blip_diffusion_pipe.text_encoder.to(dtype=torch.float16)

tsv_path = "dreambooth_prompts.tsv"
image_root = "dreambooth_single_image/dataset"
output_root = "blip_diffusion_dreambooth_42"
os.makedirs(output_root, exist_ok=True)

guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

generator = torch.Generator(device="cuda").manual_seed(42)

df = pd.read_csv(tsv_path, sep="\t")

from collections import defaultdict
counter = defaultdict(int)

for index, row in df.iterrows():
    name = row["name"]
    subject = row["subject"]
    prompt = row["prompt"]
    prompt_prefix = prompt[:30].strip().replace(" ", "_").replace("/", "_")

    image_dir = os.path.join(image_root, name)
    if not os.path.isdir(image_dir):
        print(f"Warning: image directory not found for {name}")
        continue

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.endswith("_mask.jpg")
    ]
    if not image_files:
        print(f"No valid image found for {name}")
        continue

    image_path = os.path.join(image_dir, image_files[0])
    cond_image = load_image(image_path)

    output = blip_diffusion_pipe(
        prompt,
        cond_image,
        subject,
        subject,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
        generator=generator,
    ).images

    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)
    counter[name] += 1
    filename = f"{counter[name]:03d}_{subject.replace(' ', '_')}_{prompt_prefix}.png"
    output_path = os.path.join(out_dir, filename)
    output[0].save(output_path)
    print(f"Saved: {output_path}")
