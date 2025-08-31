import os
import subprocess
import pandas as pd
from tqdm import tqdm
import shutil

DATASET_BASE_DIR = "/root/autodl-tmp/dreambooth/dataset"
PROMPT_FILE = "/root/autodl-tmp/dreambooth_prompts.tsv"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_BASE_DIR = "textual_inversion_output"
MAX_TRAIN_STEPS = 3000

def parse_prompts_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df[['name', 'subject']].drop_duplicates().values.tolist()

def train_instance(name, subject):
    instance_dir = os.path.join(DATASET_BASE_DIR, name)
    output_dir = os.path.join(OUTPUT_BASE_DIR, name)
    placeholder_token = "<_jkx1216_>"
    if placeholder_token == "<stuffed animal>":
        placeholder_token = "<_jkx1216_>"
    initializer_token = subject
    if initializer_token == "stuffed animal":
        initializer_token = "toy"
    os.makedirs(output_dir, exist_ok=True)
    
    command = [
        "accelerate", "launch", "textual_inversion_sdxl.py",
        "--pretrained_model_name_or_path", MODEL_NAME,
        "--train_data_dir", instance_dir,
        "--learnable_property", "object",
        "--placeholder_token", placeholder_token,
        "--initializer_token", initializer_token,
        "--resolution", "512",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--max_train_steps", str(MAX_TRAIN_STEPS),
        "--learning_rate", "5.0e-04",
        "--scale_lr",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--output_dir", output_dir,
        "--seed", "42",
        "--save_steps", "10000",
        "--report_to", "wandb",
        "--validation_prompt", "A photo of a <_jkx1216_>",
        "--num_validation_images", "4",
        "--validation_steps", "300",
    ]

    print(f"\n训练 {name} (subject: {subject})...")
    print(" ".join(command))
    subprocess.run(command)

    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("checkpoint"):
            print(f"delete checkpoint folder: {entry_path}")
            shutil.rmtree(entry_path)
def main():
    instances = parse_prompts_file(PROMPT_FILE)
    
    for name, subject in tqdm(instances, total=len(instances), desc="training instances"):
        print(f"\nprocessing {name} (subject: {subject})")
        train_instance(name, subject)

if __name__ == "__main__":
    main()
