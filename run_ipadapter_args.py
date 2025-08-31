import os
# os.environ["HF_DIFFUSERS_CACHE"] = "/root/autodl-tmp"
# os.environ["no_proxy"] = "localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn"
# os.environ["http_proxy"] = "http://172.20.0.113:12798"
# os.environ["https_proxy"] = "http://172.20.0.113:12798"
# os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
# os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
import copy
import cv2
import torch
import numpy as np
from pipeline_sdxl_ipadapter import StableDiffusionXLPipeline
from PIL import Image
from unet_sdxl_ip_adapter import Dift_UNet2DConditionModel
from ip_adapter.ip_adapter import IPAdapterXL, IPAdapterPlusXL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
# New added
from accelerate import Accelerator
import json
import argparse
import inversion_pipeline_sdxl_ipadapter
from diffusers.utils import load_image
from diffusers import DDIMScheduler

def resize_padding_image(image, size=224):
    old_size = image.size  # (width, height)
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.BICUBIC)
    new_image = Image.new(image.mode, (size, size), 255)
    paste_position = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    new_image.paste(image, paste_position)
    return new_image

def center_crop_resize(image, crop_ratio=0.8, size=224):
    w, h = image.size
    crop_size = int(max(w, h) * crop_ratio)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))
    return resize_padding_image(image, size)

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


def get_token_len(entity_name, tokenizer):
    entity_tokens = tokenizer(entity_name, return_tensors="pt").input_ids[0][1:-1]
    return len(entity_tokens)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Run IPAdapter with arguments")
    parser.add_argument('--base_model_path', type=str, default="stable-diffusion-xl-base-1.0/", help='Path to base model')
    parser.add_argument('--image_encoder_path', type=str, default="IP-Adapter/sdxl_models/image_encoder/", help='Path to image encoder')
    parser.add_argument('--ip_ckpt', type=str, default="IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", help='Path to IP-Adapter checkpoint')
    parser.add_argument('--output_dir', type=str, default="output/vis_sdxl/", help='Output directory')
    parser.add_argument('--scale', type=float, default=0.6, help='Scale value')
    parser.add_argument('--reference_image_path', type=str, default="assets/combination_1/cat.jpg", help='Reference image path')
    parser.add_argument('--mask_image_path', type=str, default="assets/combination_1/cat_mask.png", help='Mask image path')
    parser.add_argument('--prompt', type=str, default="a cat is running", help='Prompt for generation')
    parser.add_argument('--ref_prompt', type=str, default="a cat", help='Reference prompt')
    parser.add_argument('--seed', type=int, default=42732, help='Random seed')
    parser.add_argument('--use_ini_latents', action='store_true', help='Whether to use initial latents')
    return parser.parse_args()

args = parse_args()

base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt
output_dir = args.output_dir
scale = args.scale
reference_image_path = args.reference_image_path
mask_image_path = args.mask_image_path
prompt = args.prompt
ref_prompt = args.ref_prompt
seed = args.seed
use_ini_latents = args.use_ini_latents

num_samples = 1
num_inference_step = 50
resolution = 1024

mask = Image.open(mask_image_path).convert("L")
mask = center_crop_resize(mask, crop_ratio=1.0, size=resolution)
mask = np.array(mask)
mask = (mask > 127).astype(np.uint8)
mask = torch.from_numpy(mask).float()
mask = mask.unsqueeze(0)

image = load_image(reference_image_path)
processed_image = center_crop_resize(image, crop_ratio=1.0, size=resolution)
inversion_image = np.array(processed_image)
image = Image.open(reference_image_path).convert("RGB")   
args = parse_args()
base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt

accelerator = Accelerator()
device = "cuda"
resolution = 1024
num_inference_step = 50
prompt = "a cat is running"
ref_prompt = "a cat"

scale = args.scale
output_dir = args.output_dir

# Load model
tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer_2")

unet = Dift_UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", torch_dtype=torch.float16)

# Load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
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
print(pipe.scheduler.config.timestep_spacing)
# inversion_image = np.array(load_image(args.reference_image_path).resize((1024, 1024)))
image = load_image(args.reference_image_path)
processed_image = center_crop_resize(image, crop_ratio=1.0, size=1024)
inversion_image = np.array(processed_image)
# image = Image.open(args.reference_image_path).resize((512, 512))
image = Image.open(args.reference_image_path).convert("RGB")   
image = center_crop_resize(image, crop_ratio=1.0, size=resolution)
cur_model = IPAdapterXL
if ip_ckpt is None:
    ip_model = cur_model(pipe, image_encoder_path, ip_ckpt=None, device=device)
else:
    ip_model = cur_model(pipe, image_encoder_path, ip_ckpt=ip_ckpt, device=device)
os.makedirs(output_dir, exist_ok=True)

skip = .1

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = ip_model.get_embeddings(pil_image=image, prompt=ref_prompt, num_samples=num_samples, scale=scale)
ref_latents = inversion_pipeline_sdxl_ipadapter.ddim_inversion(pipe, inversion_image, ref_prompt, num_inference_step, 7.5, scale, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, skip = skip) #[0,1,2,...,50]
ref_dift = pipe.unet.dift_latent_store.dift_features
ref_dift = {
    k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    for k, v in ref_dift.items()
}

feature_merged_time = None
_lambda = 0.5

generated_images, dift_feature = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=num_inference_step, seed=seed, prompt=prompt, scale=scale, ref_dift=ref_dift, feature_merged_time=feature_merged_time, _lambda=_lambda, ref_latents=ref_latents, mask=mask, use_ini_latents = use_ini_latents, skip = skip)
for idx, save_image in enumerate(generated_images):
    save_image.save(os.path.join(output_dir, '{}_gen_by_original_model.png'.format(prompt)))


dift_feature = pipe.unet.dift_latent_store.dift_features

cur_model = IPAdapterXL
merged_layers = [x for x in range(44, 71)]
merged_times = [x for x in range(20, 51)]

ip_model.set_dift_ipadapter(ref_dift=ref_dift, tgt_dift=dift_feature, merged_layers=merged_layers, merged_times=merged_times, ref_mask=mask)

generated_images = ip_model.generate_with_dift(pil_image=image, num_samples=num_samples, num_inference_steps=num_inference_step, seed=seed, prompt=prompt, ref_prompt=ref_prompt, scale=scale, ref_dift=ref_dift, feature_merged_time=feature_merged_time, _lambda=_lambda, ref_latents=ref_latents, mask=mask, target_dift=dift_feature, use_ini_latents = use_ini_latents, skip = skip)
for idx, save_image in enumerate(generated_images):
    save_image.save(os.path.join(output_dir, 'ours_gen_{}_{}_{}.png'.format(idx, merged_layers[0], merged_times[0])))
    print("save image to {}".format(os.path.join(output_dir, 'ours_gen_{}_{}_{}.png'.format(idx, merged_layers[0], merged_times[0]))))
    break