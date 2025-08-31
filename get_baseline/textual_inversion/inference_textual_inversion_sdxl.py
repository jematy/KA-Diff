from diffusers import StableDiffusionXLPipeline
import torch
from safetensors.torch import load_file

file = "/root/autodl-tmp/textual_inversion_output/cat2/learned_embeds.safetensors"
state_dict = load_file(file)

file1 = "/root/autodl-tmp/textual_inversion_output/cat2/learned_embeds_2.safetensors"
state_dict1 = load_file(file1)

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_auth_token=True).to("cuda")
pipe.to("cuda")

pipe.load_textual_inversion(state_dict1["<_jkx1216_>"], token="<_jkx1216_>", text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
pipe.load_textual_inversion(state_dict["<_jkx1216_>"], token="<_jkx1216_>", text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

generator = torch.Generator().manual_seed(42)
image = pipe("a <_jkx1216_>", generator=generator).images[0]
image.save("backpack.png")
