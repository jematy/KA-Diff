import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def get_image_latent(
    x0: np.ndarray,
    pipe: StableDiffusionPipeline
) -> torch.Tensor:
    """
    Encode an image to latent space using Stable Diffusion pipeline.

    Args:
        x0: Input image as H×W×3 uint8 numpy array.
        pipe: Initialized StableDiffusionPipeline.

    Returns:
        A tensor of latent representation with shape [1, C, H/8, W/8].
    """
    img = torch.from_numpy(x0).float() / 255.
    img = img.half()
    img = (img * 2 - 1).permute(2, 0, 1).unsqueeze(0).to(pipe.device)
    latents = pipe.vae.encode(img).latent_dist.mean * pipe.vae.config.scaling_factor
    return latents

@torch.no_grad()
def ddim_inversion_v1_5(
    pipe: StableDiffusionPipeline,
    x0: np.ndarray,
    prompt: str,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
) -> torch.Tensor:
    """
    Perform DDIM inversion on an input image using Stable Diffusion v1.5 pipeline.

    Args:
        x0: Input image as H×W×3 uint8 numpy array.
        prompt: Positive text prompt.
        negative_prompt: Negative text prompt.
        pipe: Initialized StableDiffusionPipeline for "runwayml/stable-diffusion-v1-5".
        num_inference_steps: Number of DDIM steps.
        guidance_scale: Classifier-free guidance scale.

    Returns:
        A tensor of latent sequences with shape [steps+1, 1, C, H/8, W/8].
    """
    # Encode image to latent
    device = pipe.device

    start_latents = get_image_latent(x0, pipe)
    
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    intermediate_latents.append(latents)

    for i in tqdm(range(pipe.scheduler.num_inference_steps)):

        t = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) - i - 1]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents).flip(0)

@torch.no_grad()
def ddim_inversion_v1_5_ipadapter(
    pipe: StableDiffusionPipeline,
    x0: np.ndarray,
    prompt_embeds: str,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt_embeds="",
) -> torch.Tensor:
    """
    Perform DDIM inversion on an input image using Stable Diffusion v1.5 pipeline.

    Args:
        x0: Input image as H×W×3 uint8 numpy array.
        prompt: Positive text prompt.
        negative_prompt: Negative text prompt.
        pipe: Initialized StableDiffusionPipeline for "runwayml/stable-diffusion-v1-5".
        num_inference_steps: Number of DDIM steps.
        guidance_scale: Classifier-free guidance scale.

    Returns:
        A tensor of latent sequences with shape [steps+1, 1, C, H/8, W/8].
    """
    # Encode image to latent
    device = pipe.device

    start_latents = get_image_latent(x0, pipe)
    
    if do_classifier_free_guidance:
        text_embeddings = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    else:
        text_embeddings = prompt_embeds
        
    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    intermediate_latents.append(latents)

    for i in tqdm(range(pipe.scheduler.num_inference_steps)):

        t = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) - i - 1]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents).flip(0)
