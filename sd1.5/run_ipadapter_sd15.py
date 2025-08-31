import os
os.environ["HF_DIFFUSERS_CACHE"] = "/root/autodl-tmp"
os.environ["no_proxy"] = "localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn"
os.environ["http_proxy"] = "http://172.20.0.113:12798"
os.environ["https_proxy"] = "http://172.20.0.113:12798"
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
import copy
import cv2
import torch
import numpy as np
from pipeline_sd15 import StableDiffusionPipeline
from PIL import Image
from unet_sd15_ip_adapter import Dift_UNet2DConditionModel
from ip_adapter.ip_adapter import IPAdapter
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
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
    class Args:
        # base_model_path = "/root/autodl-tmp/stable-diffusion-v1-5/"
        base_model_path = "/root/autodl-tmp/Realistic_Vision_V4.0_noVAE/"
        image_encoder_path = "/root/autodl-tmp/IP-Adapter/models/image_encoder/"
        ip_ckpt = "/root/autodl-tmp/IP-Adapter/models/ip-adapter_sd15.bin"
        output_dir = "/root/autodl-tmp/mip-adapter_finegrain/output/vis_sd15/"
        scale = 0.6
        reference_image1_path = "/root/autodl-tmp/mip-adapter_finegrain/assets/combination_1/cat.jpg"
        # reference_image2_path = "/root/autodl-tmp/mip-adapter_finegrain/assets/combination_1/jacket.jpg"
        # reference_image1_path = "/root/autodl-tmp/mip-adapter_finegrain/assets/lexus_car.png"
        # reference_image1_path = "/root/autodl-tmp/mip-adapter_finegrain/assets/Spree.png"
        # prompt = "a cat is eating with the background of boston city"
        num_samples = 1
        is_plus = False

    return Args()

mask_image = "/root/autodl-tmp/mip-adapter_finegrain/assets/combination_1/cat_mask.png"
# mask_image = "/root/autodl-tmp/mip-adapter_finegrain/assets/lexus_car_mask.png"
# mask_image = "/root/autodl-tmp/mip-adapter_finegrain/assets/Spree_mask.png"
mask = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE)
mask = (mask > 127).astype(np.uint8)
mask = torch.from_numpy(mask).float()
mask = mask.unsqueeze(0)

args = parse_args()
base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt

accelerator = Accelerator()
device = "cuda"
num_inference_step = 50
prompt = "a cat is running"
# prompt = "a lexus car is parked near a lake"
# prompt = "A 1986 Honda Spree parked on a sunny beach boardwalk"

ref_prompt = "a cat"
# ref_prompt = "a car"
# ref_prompt = "A 1986 Honda Spree"
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
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt=ip_ckpt, device=device,model_type="sd15")
os.makedirs(output_dir, exist_ok=True)

prompt_embeds, negative_prompt_embeds = ip_model.get_sd15_embeddings(pil_image=image, prompt=ref_prompt, negative_prompt="", scale=scale, num_samples=1)
ref_latents = inversion_pipeline_sd15_ipadapter.ddim_inversion_v1_5_ipadapter(pipe = pipe, x0 = inversion_image, prompt_embeds = prompt_embeds, num_inference_steps = num_inference_step, guidance_scale = 7.5, negative_prompt_embeds = negative_prompt_embeds)
ref_dift = pipe.unet.dift_latent_store.dift_features
ref_dift = {
    k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    for k, v in ref_dift.items()
}


# feature_merged_time = [x for x in range(20, 51)]
feature_merged_time = None
_lambda = 0.5

generated_images = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=num_inference_step, seed=2, prompt=prompt, scale=scale, ref_dift=ref_dift, feature_merged_time=feature_merged_time, _lambda=_lambda, ref_latents=ref_latents, mask=mask, use_ini_latents = True)
for idx, save_image in enumerate(generated_images):
    save_image.save(os.path.join(output_dir, '{}_gen_{}.png'.format(prompt, idx)))

# # _, dift_feature = dift_feature.chunk(2)
# # _, ref_dift = ref_dift.chunk(2)
# dift_feature = {
#     k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
#     for k, v in dift_feature.items()
# }

dift_feature = pipe.unet.dift_latent_store.dift_features

cur_model = IPAdapter
merged_layers = [x for x in range(9, 17)]  # 9 layers each block
merged_times = [x for x in range(20, 51)]

# if ip_ckpt is None:
#     ip_model = cur_model(pipe, image_encoder_path, ip_ckpt=None, device=device, ref_dift=ref_dift, tgt_dift=dift_feature, merged_layers=merged_layers, merged_times=merged_times, ref_mask=mask)
# else:
#     ip_model = cur_model(pipe, image_encoder_path, ip_ckpt=ip_ckpt, device=device, ref_dift=ref_dift, tgt_dift=dift_feature, merged_layers=merged_layers, merged_times=merged_times, ref_mask=mask)

ip_model.set_dift_ipadapter(ref_dift=ref_dift, tgt_dift=dift_feature, merged_layers=merged_layers, merged_times=merged_times, ref_mask=mask)

generated_images = ip_model.generate_with_dift(pil_image=image, num_samples=num_samples, num_inference_steps=num_inference_step, seed=2, prompt=prompt, ref_prompt=ref_prompt, scale=scale, ref_dift=ref_dift, feature_merged_time=feature_merged_time, _lambda=_lambda, ref_latents=ref_latents, mask=mask, target_dift=dift_feature, use_ini_latents = True)
for idx, save_image in enumerate(generated_images):
    save_image.save(os.path.join(output_dir, 'Our_{}_gen_{}_{}_{}_relastic.png'.format(prompt, idx, merged_layers[0], merged_times[0])))
    print("save image to {}".format(os.path.join(output_dir, 'Our_{}_gen_{}_{}_{}.png'.format(prompt, idx, merged_layers[0], merged_times[0]))))