from diffusers import DiffusionPipeline

import torch

import os
os.environ["HF_HOME"] = "/data/sdxl"


base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

lora_weights_path = "/data/diffusers/examples/dreambooth/lora-trained-xl" 

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)

pipe = pipe.to("cuda")


pipe.load_lora_weights(lora_weights_path)


image = pipe("a sks dog is running with the background of city", num_inference_steps=50, seed = 4).images[0]

image.save("sks_dog.png")