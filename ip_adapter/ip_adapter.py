import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

from .attention_processor import AttnProcessor, IPAttnProcessor
from .resampler import Resampler

import random
import numpy as np

def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, ref_dift=None, tgt_dift=None, merged_layers=None, merged_times=None, ref_mask=None, model_type="SDXL"):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)

        if model_type == "SDXL":
            self.set_ip_adapter(ref_dift, tgt_dift, merged_layers, merged_times, ref_mask)
        else:
            self.set_ip_adapter_SD15(ref_dift, tgt_dift, merged_layers, merged_times, ref_mask)
        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self, ref_dift=None, tgt_dift=None, merged_layers=None, merged_times=None, ref_mask=None):
        unet = self.pipe.unet
        attn_procs = {}
        cnt = 0
        cnt1 = 0
        cnt2 = 0
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
                if cross_attention_dim is None:
                    cnt1 += 1
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                if cross_attention_dim is None:
                    cnt2 += 1
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
                if cross_attention_dim is None:
                    cnt += 1
            if cross_attention_dim is None:
                if name.startswith("mid_block"):
                    attn_procs[name] = AttnProcessor(attn_layer = 24+cnt1, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)
                if name.startswith("up_blocks"):
                    attn_procs[name] = AttnProcessor(attn_layer = 34+cnt2, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)
                if name.startswith("down_blocks"):
                    attn_procs[name] = AttnProcessor(attn_layer = cnt, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)
            else:
                if name.startswith("mid_block"):
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        attn_layer = 24+cnt1, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask
                    ).to(self.device, dtype=torch.float16)
                if name.startswith("up_blocks"):
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        attn_layer = 34+cnt2, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask
                    ).to(self.device, dtype=torch.float16)
                if name.startswith("down_blocks"):
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        attn_layer = cnt, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

    def set_ip_adapter_SD15(self, ref_dift=None, tgt_dift=None, merged_layers=None, merged_times=None, ref_mask=None):
        unet = self.pipe.unet
        attn_procs = {}
        cnt = 0
        cnt1 = 0
        cnt2 = 0
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
                if cross_attention_dim is None:
                    cnt1 += 1   #1
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                if cross_attention_dim is None:
                    cnt2 += 1   #9
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
                if cross_attention_dim is None:
                    cnt += 1    #6
            if cross_attention_dim is None:
                if name.startswith("mid_block"):
                    attn_procs[name] = AttnProcessor(attn_layer = 6+cnt1, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)
                if name.startswith("up_blocks"):
                    attn_procs[name] = AttnProcessor(attn_layer = 7+cnt2, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)
                if name.startswith("down_blocks"):
                    attn_procs[name] = AttnProcessor(attn_layer = cnt, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)
            else:
                if name.startswith("mid_block"):
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        attn_layer = 24+cnt1, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask
                    ).to(self.device, dtype=torch.float16)
                if name.startswith("up_blocks"):
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        attn_layer = 34+cnt2, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask
                    ).to(self.device, dtype=torch.float16)
                if name.startswith("down_blocks"):
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.num_tokens,
                        attn_layer = cnt, ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask
                    ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
    

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def apply_masked_patch(self, ini_latents, ref_latents, mask, target_size=(128, 128), threshold=0.5):
        import torch.nn.functional as F
        dtype = ini_latents.dtype
        device = ini_latents.device

        # 1. resize mask
        resized_mask = F.interpolate(mask.unsqueeze(0).to(dtype=dtype, device=device), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        # 2. bool mask
        bool_mask = resized_mask > threshold  # shape: [1, H, W], dtype: bool
        bool_mask_4ch = bool_mask.expand_as(ini_latents[0])
        #output the number of True
        num_true = bool_mask_4ch.sum().item()

        # 3. apply mask
        ini_latents[0][bool_mask_4ch] = ref_latents[0].to(dtype=dtype, device=device)[bool_mask_4ch]

        return ini_latents

    def set_dift_ipadapter(self, ref_dift=None, tgt_dift=None, merged_layers=None, merged_times=None, ref_mask=None):
        for name, proc in self.pipe.unet.attn_processors.items():
            if isinstance(proc, AttnProcessor) or isinstance(proc, IPAttnProcessor):
                 proc.add_dift(ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask, up_index=1)

    def get_sd15_embeddings(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        return prompt_embeds, negative_prompt_embeds

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        ref_dift = None,
        feature_merged_time=None,
        _lambda=None,
        ref_latents=None,
        mask=None,
        target_dift=None,
        use_ini_latents=False,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        ini_latents = torch.randn(1, 4, 64, 64, generator=self.generator, device=self.device, dtype=torch.float16)

        if use_ini_latents:
            ini_latents[0] = self.apply_masked_patch(ini_latents, ref_latents, mask, target_size=(64, 64), threshold=0.5)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            ref_dift=ref_dift,
            feature_merged_time=feature_merged_time,
            _lambda=_lambda,
            ref_latents=None,
            mask=mask,
            latents=ini_latents,
            **kwargs,
        ).images

        return images

    def generate_with_dift(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        ref_dift = None,
        feature_merged_time=None,
        _lambda=None,
        ref_latents=None,
        mask=None,
        target_dift=None,
        use_ini_latents=False,
        ref_prompt="",
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        with torch.inference_mode():
            ref_prompt_embeds, ref_negative_prompt_embeds = self.pipe.encode_prompt(
                [ref_prompt],
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            ref_prompt_embeds = torch.cat([ref_prompt_embeds, image_prompt_embeds], dim=1)
            ref_negative_prompt_embeds = torch.cat([ref_negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        ini_latents = torch.randn(2, 4, 64, 64, generator=self.generator, device=self.device, dtype=torch.float16)

        if use_ini_latents:
            print("use_ini_latents")
            ini_latents = self.apply_masked_patch(ini_latents, ref_latents, mask, target_size=(64, 64), threshold=0.5)

        #if two images, concatenate the prompt_embeds and ref_prompt_embeds
        prompt_embeds = torch.cat([prompt_embeds, ref_prompt_embeds], dim=0)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, ref_negative_prompt_embeds], dim=0)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            ref_dift=ref_dift,
            feature_merged_time=feature_merged_time,
            _lambda=_lambda,
            ref_latents=ref_latents,
            mask=mask,
            latents=ini_latents,
            **kwargs,
        ).images

        return images
    
class IPAdapterXL(IPAdapter):
    """SDXL"""

    def get_embeddings(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None
    ):
        if seed is not None:
            set_seed_everywhere(seed)
        self.set_scale(scale)

        num_prompts = 1# if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    def apply_masked_patch(self, ini_latents, ref_latents, mask, target_size=(128, 128), threshold=0.5):
        import torch.nn.functional as F
        dtype = ini_latents.dtype
        device = ini_latents.device

        # 1. resize mask
        resized_mask = F.interpolate(mask.unsqueeze(0).to(dtype=dtype, device=device), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        # 2. bool mask
        bool_mask = resized_mask > threshold  # shape: [1, H, W], dtype: bool
        bool_mask_4ch = bool_mask.expand_as(ini_latents[0])

        # 3. apply mask
        ini_latents[0][bool_mask_4ch] = ref_latents[0].to(dtype=dtype, device=device)[bool_mask_4ch]

        return ini_latents
    
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        ref_dift = None,
        feature_merged_time=None,
        _lambda=None,
        ref_latents=None,
        mask=None,
        target_dift=None,
        use_ini_latents=False,
        **kwargs,
    ):
        if seed is not None:
            set_seed_everywhere(seed)
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        ini_latents = torch.randn(1, 4, 128, 128, generator=self.generator, device=self.device, dtype=torch.float16)

        # ini_latents[0] = ref_latents[0].unsqueeze(0)
        if use_ini_latents:
            ini_latents[0] = self.apply_masked_patch(ini_latents, ref_latents, mask, target_size=(128, 128), threshold=0.5)

        # print(ini_latents)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            latents=ini_latents,
            **kwargs,
        ).images
        # print(self.pipe.unet.dift_latent_store.dift_features.keys())
        return images, self.pipe.unet.dift_latent_store.dift_features

    def set_dift_ipadapter(self, ref_dift=None, tgt_dift=None, merged_layers=None, merged_times=None, ref_mask=None):
        for name, proc in self.pipe.unet.attn_processors.items():
            if isinstance(proc, AttnProcessor) or isinstance(proc, IPAttnProcessor):
                 proc.add_dift(ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)

    def generate_with_dift(
        self,
        pil_image,
        prompt=None,
        ref_prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        ref_dift = None,
        feature_merged_time=None,
        _lambda=None,
        ref_latents=None,
        mask=None,
        target_dift=None,
        use_ini_latents=False,
        **kwargs,
    ):
        if seed is not None:
            set_seed_everywhere(seed)
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        with torch.inference_mode():
            (
                ref_prompt_embeds,
                ref_negative_prompt_embeds,
                ref_pooled_prompt_embeds,
                ref_negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                ref_prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            ref_prompt_embeds = torch.cat([ref_prompt_embeds, image_prompt_embeds], dim=1)
            ref_negative_prompt_embeds = torch.cat([ref_negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        ini_latents = torch.randn(2, 4, 128, 128, generator=self.generator, device=self.device, dtype=torch.float16)
        # ini_latents[0] = ref_latents[0].unsqueeze(0)
        if use_ini_latents:
            ini_latents = self.apply_masked_patch(ini_latents, ref_latents, mask, target_size=(128, 128), threshold=0.5)

        #if two images, concatenate the prompt_embeds and ref_prompt_embeds
        prompt_embeds = torch.cat([prompt_embeds, ref_prompt_embeds], dim=0)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, ref_negative_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, ref_pooled_prompt_embeds], dim=0)
        negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, ref_negative_pooled_prompt_embeds], dim=0)

        # print(ini_latents)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            ref_dift=ref_dift,
            feature_merged_time=feature_merged_time,
            _lambda=_lambda,
            ref_latents=ref_latents,
            mask=mask,
            target_dift=target_dift,
            latents=ini_latents,
            **kwargs,
        ).images
        return images
    
class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def get_embeddings(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None
    ):
        if seed is not None:
            set_seed_everywhere(seed)
        self.set_scale(scale)

        num_prompts = 1# if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        ref_dift = None,
        feature_merged_time=None,
        _lambda=None,
        ref_latents=None,
        mask=None,
        target_dift=None,
        use_ini_latents=False,
        skip = .15,
        **kwargs,
    ):
        if seed is not None:
            set_seed_everywhere(seed)
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        ini_latents = torch.randn(1, 4, 128, 128, generator=self.generator, device=self.device, dtype=torch.float16)

        # ini_latents[0] = ref_latents[0].unsqueeze(0)
        if use_ini_latents:
            ini_latents[0] = self.apply_masked_patch(ini_latents, ref_latents, mask, target_size=(128, 128), threshold=0.5)

        # print(ini_latents)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            latents=ini_latents,
            skip = skip,
            **kwargs,
        ).images
        # print(self.pipe.unet.dift_latent_store.dift_features.keys())
        return images, self.pipe.unet.dift_latent_store.dift_features

    def set_dift_ipadapter(self, ref_dift=None, tgt_dift=None, merged_layers=None, merged_times=None, ref_mask=None):
        for name, proc in self.pipe.unet.attn_processors.items():
            if isinstance(proc, AttnProcessor) or isinstance(proc, IPAttnProcessor):
                 proc.add_dift(ref_dift=ref_dift, tgt_dift=tgt_dift, merged_layers=merged_layers, merged_times=merged_times, ref_mask=ref_mask)

    def generate_with_dift(
        self,
        pil_image,
        prompt=None,
        ref_prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        ref_dift = None,
        feature_merged_time=None,
        _lambda=None,
        ref_latents=None,
        mask=None,
        target_dift=None,
        use_ini_latents=False,
        skip = .15,
        **kwargs,
    ):
        if seed is not None:
            set_seed_everywhere(seed)
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        with torch.inference_mode():
            (
                ref_prompt_embeds,
                ref_negative_prompt_embeds,
                ref_pooled_prompt_embeds,
                ref_negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                ref_prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            ref_prompt_embeds = torch.cat([ref_prompt_embeds, image_prompt_embeds], dim=1)
            ref_negative_prompt_embeds = torch.cat([ref_negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        ini_latents = torch.randn(2, 4, 128, 128, generator=self.generator, device=self.device, dtype=torch.float16)
        # ini_latents[0] = ref_latents[0].unsqueeze(0)
        if use_ini_latents:
            ini_latents = self.apply_masked_patch(ini_latents, ref_latents, mask, target_size=(128, 128), threshold=0.5)

        #if two images, concatenate the prompt_embeds and ref_prompt_embeds
        prompt_embeds = torch.cat([prompt_embeds, ref_prompt_embeds], dim=0)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, ref_negative_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, ref_pooled_prompt_embeds], dim=0)
        negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, ref_negative_pooled_prompt_embeds], dim=0)

        # print(ini_latents)
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            ref_dift=ref_dift,
            feature_merged_time=feature_merged_time,
            _lambda=_lambda,
            ref_latents=ref_latents,
            mask=mask,
            target_dift=target_dift,
            latents=ini_latents,
            skip = skip,
            **kwargs,
        ).images
        return images