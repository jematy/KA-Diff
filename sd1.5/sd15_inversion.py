import os
os.environ["HF_DIFFUSERS_CACHE"] = "/root/autodl-tmp"
os.environ["no_proxy"] = "localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn"
os.environ["http_proxy"] = "http://172.20.0.113:12798"
os.environ["https_proxy"] = "http://172.20.0.113:12798"
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

import cv2
import torch
import numpy as np
from pipeline_sd15 import StableDiffusionPipeline
from PIL import Image
from dift_unet_sdxl import Dift_UNet2DConditionModel
from ip_adapter.ip_adapter import IPAdapter
from transformers import CLIPTextModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
# New added
from accelerate import Accelerator
import json
import argparse
import inversion_pipeline_sd15_ipadapter
from diffusers.utils import load_image
from diffusers import DDIMScheduler

def collate_fn(data):
    # New added
    comb_idxes = [example["comb_idx"] for example in data]
    prompts = [example["prompt"] for example in data]
    prompt_token_lens = [example["prompt_token_len"] for example in data]
    entity_names = [example["entity_names"] for example in data]
    clip_images, entity_indexes = [], []
    for example in data:
        clip_images.extend([example["entity_imgs"][0], example["entity_imgs"][1]])
        entity_indexes.append(example["entity_indexes"])

    return {
        "comb_idxes": comb_idxes,
        "clip_images": clip_images,
        "prompts": prompts,
        "prompt_token_lens": prompt_token_lens,
        "entity_names": entity_names,
        "entity_indexes": entity_indexes,
    }


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def recover_image(img_tensor, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    mean = torch.FloatTensor(mean).cuda() if img_tensor.device.type == 'cuda' else torch.FloatTensor(mean)
    std = torch.FloatTensor(std).cuda() if img_tensor.device.type == 'cuda' else torch.FloatTensor(std)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).mul(255).cpu().byte().numpy()
    img_pil = Image.fromarray(img_np, 'RGB')

    return img_pil




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# def parse_args():
#     parser = argparse.ArgumentParser("metric", add_help=False)
#     parser.add_argument("--base_model_path", type=str)
#     parser.add_argument("--image_encoder_path", type=str)
#     parser.add_argument("--ip_ckpt", type=str)
#     parser.add_argument("--output_dir", type=str)
#     parser.add_argument("--scale", type=float, default=0.6)
#     parser.add_argument("--reference_image1_path", type=str)
#     parser.add_argument("--reference_image2_path", type=str)
#     parser.add_argument("--prompt", type=str)
#     parser.add_argument("--num_samples", type=int, default=1)
#     parser.add_argument("--is_plus", type=str2bool, default=False)
#     return parser.parse_args()

def parse_args():
    class Args:
        base_model_path = "/root/autodl-tmp/stable-diffusion-v1-5/"
        image_encoder_path = "/root/autodl-tmp/IP-Adapter/models/image_encoder/"
        ip_ckpt = "/root/autodl-tmp/IP-Adapter/models/ip-adapter_sd15.bin"
        output_dir = "/root/autodl-tmp/mip-adapter_finegrain/output/vis_sd15/"
        scale = 0.75
        reference_image1_path = "/root/autodl-tmp/mip-adapter_finegrain/assets/combination_1/cat.jpg"
        reference_image2_path = "/root/autodl-tmp/mip-adapter_finegrain/assets/combination_1/jacket.jpg"
        prompt = "A wide shot of cat wearing jacket with boston city in background."
        num_samples = 1
        is_plus = False

    return Args()

args = parse_args()
base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt

accelerator = Accelerator()
device = "cuda"
resolution = 512
num_tokens = 4 if not args.is_plus else 16
num_objects = 2
num_inference_step = 50

state_dict = None
num_samples = args.num_samples
scale = args.scale
output_dir = args.output_dir

unet = Dift_UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", torch_dtype=torch.float16)

# Load SDXL pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    unet=unet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_slicing()
pipe.to(device)

pipe.scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False)

inversion_image = np.array(load_image(args.reference_image1_path).resize((512, 512)))

image = Image.open(args.reference_image1_path).resize((512, 512))

import copy

ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt=ip_ckpt, device=device)

prompt_embeds, negative_prompt_embeds = ip_model.get_sd15_embeddings(pil_image=image, prompt="a cat", negative_prompt="", scale=0.0, num_samples=1)
ref_latents = inversion_pipeline_sd15_ipadapter.ddim_inversion_v1_5_ipadapter(pipe = pipe, x0 = inversion_image, prompt_embeds = prompt_embeds, num_inference_steps = num_inference_step, guidance_scale = 7.5, negative_prompt_embeds = negative_prompt_embeds)

import os
from torchvision.utils import save_image

save_dir = os.path.join(output_dir, "ref_latents_images")
os.makedirs(save_dir, exist_ok=True)

scaling_factor = pipe.vae.config.scaling_factor

# 把每个latent变成图
for i, latent in enumerate(ref_latents):
    latent = latent.unsqueeze(0)  # VAE decode expects batch dimension
    latent = latent.to(dtype=torch.float32, device=pipe.vae.device)
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    with torch.no_grad():
        image = pipe.vae.decode(latent / scaling_factor).sample

    # Post-process to [0,1]
    image = (image / 2 + 0.5).clamp(0, 1)

    # Save image
    save_path = os.path.join(save_dir, f"{i:03d}.png")
    save_image(image, save_path)

    print(f"Saved {save_path}")