import os
import torch
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterPlusXL
from transformers import CLIPTokenizer
from tqdm import tqdm
os.environ["HF_HOME"] = "/root/autodl-tmp"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp"
os.environ["DIFFUSERS_CACHE"] = "/root/autodl-tmp"

def resize_padding_image(image, size=224):
    old_size = image.size  # (width, height)
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.BICUBIC)
    new_image = Image.new("RGB", (size, size), (255, 255, 255))
    paste_position = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    new_image.paste(image, paste_position)
    return new_image

def center_crop_resize(image, crop_ratio=0.8, size=224):
    w, h = image.size
    crop_size = int(min(w, h) * crop_ratio)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))
    return resize_padding_image(image, size)

base_model_path = "SG161222/RealVisXL_V1.0"
image_encoder_path = "/root/autodl-tmp/IP-Adapter/models/image_encoder/"
ip_ckpt = "/root/autodl-tmp/Patch_model/PatchDPO/model.bin"
tsv_path = "/root/autodl-tmp/PatchDPO/prompt.tsv"
ref_image_folder = "/root/autodl-tmp/test_dataset"
output_folder = "Pest_seed42"
num_tokens = 16
scale = 0.6
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(output_folder, exist_ok=True)

tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_slicing()
pipe.to(device)

ip_model = IPAdapterPlusXL(pipe, image_encoder_path, None, ip_ckpt, device, num_tokens=num_tokens)

df = pd.read_csv(tsv_path, sep='\t')

for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
    index = row['index']
    prompt = row['prompt']
    image_path = os.path.join(ref_image_folder, f"{index}.jpg")
    
    if not os.path.exists(image_path):
        print(f"warning: image {image_path} not found, skip")
        continue
    
    raw_image = Image.open(image_path).convert("RGB")
    input_image = center_crop_resize(raw_image, crop_ratio=0.8, size=224)

    try:
        generated_images = ip_model.generate_multi(
            pil_image=[[input_image]],  # note: nested list
            prompt=[prompt],
            num_samples=1,
            num_inference_steps=50,
            seed=42,
            scale=scale,
        )

        save_path = os.path.join(output_folder, f"{index}_0.png")
        generated_images[0].save(save_path)
        print(f"save image to {save_path}")
    except Exception as e:
        print(f"error when processing index {index}: {e}")
