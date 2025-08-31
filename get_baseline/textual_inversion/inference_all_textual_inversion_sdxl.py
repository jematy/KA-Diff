import os
import pandas as pd
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

base_model = "stabilityai/stable-diffusion-xl-base-1.0"

df = pd.read_csv("/root/autodl-tmp/dreambooth_prompts.tsv", sep='\t')

output_base = "/root/autodl-tmp/textual_inversion_dreambooth_42"

for name, group in df.groupby("name"):

    subject = group.iloc[0]["subject"]
    placeholder_token = "<_jkx1216_>"

    token_dir = os.path.join("/root/autodl-tmp/textual_inversion_output", name)
    file     = os.path.join(token_dir, "learned_embeds.safetensors")
    file1    = os.path.join(token_dir, "learned_embeds_2.safetensors")

    state_dict  = load_file(file)
    state_dict1 = load_file(file1)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.load_textual_inversion(
        state_dict1[placeholder_token],
        token=placeholder_token,
        text_encoder=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer_2
    )
    pipe.load_textual_inversion(
        state_dict[placeholder_token],
        token=placeholder_token,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer
    )

    output_dir = os.path.join(output_base, name)
    os.makedirs(output_dir, exist_ok=True)

    for i, row in group.reset_index(drop=True).iterrows():
        prompt = row["prompt"]
        prompt_with_token = prompt.replace(subject, placeholder_token)

        generator = torch.Generator().manual_seed(42)
        image = pipe(
            prompt_with_token,
            num_inference_steps=50,
            generator=generator
        ).images[0]

        clean_prompt = prompt.replace(" ", "_")
        filename = f"{i+1:03d}_{subject.replace(' ', '_')}_{clean_prompt}.png"
        filepath = os.path.join(output_dir, filename)

        image.save(filepath)
        print(f"Saved: {filepath}")
