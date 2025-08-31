import os
import pandas as pd
from diffusers.pipelines.blip_diffusion.pipeline_blip_diffusion import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch
from PIL import Image

# Initialize model
blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "Salesforce/blipdiffusion", torch_dtype=torch.float16, ignore_mismatched_sizes=True
).to("cuda")
blip_diffusion_pipe.text_encoder.to(dtype=torch.float16)

# Path settings
tsv_path = "/root/autodl-tmp/prompt.tsv"
image_root = "/root/autodl-tmp/test_dataset"
output_root = "/root/autodl-tmp/blip_diffusion_our_dataset_42"
os.makedirs(output_root, exist_ok=True)

# Inference settings
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

generator = torch.Generator(device="cuda").manual_seed(42)

# Read tsv
df = pd.read_csv(tsv_path, sep="\t")

# Iterate over each row
for index, row in df.iterrows():
    image_index = row["index"]
    prompt = row["prompt"]
    subject = row["subject"]

    image_path = os.path.join(image_root, f"{image_index}.jpg")
    if not os.path.isfile(image_path):
        print(f"Warning: image not found for index {image_index}")
        continue

    cond_image = load_image(image_path)

    # Generate image
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

    # Save image
    output_path = os.path.join(output_root, f"{image_index}_0.png")
    output[0].save(output_path)
    print(f"Saved: {output_path}")
