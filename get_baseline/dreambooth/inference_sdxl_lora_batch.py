from diffusers import DiffusionPipeline
import torch
import os
import csv
import argparse
from pathlib import Path
import re

def clean_filename(text):
    text = re.sub(r'[\\/*?:"<>|]', '', text)
    text = text.replace(' ', '_')
    return text

os.environ["HF_HOME"] = "/data/sdxl"

parser = argparse.ArgumentParser(description='batch generate SDXL LoRA images')
parser.add_argument('--seed', type=int, default=42, help='random seed for generation')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps')
parser.add_argument('--tsv_file', type=str, default='/data/dreambooth_prompts.tsv', help='TSV file path')
parser.add_argument('--lora_dir', type=str, default='/data/lora-trained-dreambooth-xl', help='LoRA model directory')
parser.add_argument('--output_dir', type=str, default='/data/generated_images', help='output image directory')
parser.add_argument('--names', nargs='+', help='only process the specified name list, if not specified, process all')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

data = []
with open(args.tsv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    headers = next(reader)
    for row in reader:
        if len(row) >= 3:
            name, subject, prompt = row[0], row[1], row[2]
            if args.names and name not in args.names:
                continue
            data.append((name, subject, prompt))

name_groups = {}
for name, subject, prompt in data:
    if name not in name_groups:
        name_groups[name] = []
    name_groups[name].append((subject, prompt))

for name, prompts in name_groups.items():
    lora_weights_path = os.path.join(args.lora_dir, name)
    
    if not os.path.exists(lora_weights_path):
        print(f"warning: {name} LoRA weights not found in {lora_weights_path}, skip")
        continue
    
    print(f"\nprocess {name} with {len(prompts)} prompts...")
    print(f"load LoRA weights: {lora_weights_path}")
    
    pipe.load_lora_weights(lora_weights_path)
    
    for i, (subject, prompt) in enumerate(prompts):
        modified_prompt = prompt.replace(subject, f"sks {subject}", 1)
        
        output_subdir = os.path.join(args.output_dir, name)
        os.makedirs(output_subdir, exist_ok=True)

        prompt_prefix = prompt[:40] if len(prompt) > 40 else prompt
        prompt_prefix = clean_filename(prompt_prefix)

        output_filename = f"{i+1:03d}_{subject.replace(' ', '_')}_{prompt_prefix}.png"
        output_path = os.path.join(output_subdir, output_filename)
        
        print(f"  generate {name} - prompt: {prompt}")
        
        image = pipe(modified_prompt, num_inference_steps=args.steps, seed=args.seed).images[0]
        
        image.save(output_path)
        print(f"  save image to {output_path}")
    
    pipe.unload_lora_weights()

print("\n done!")