import os
import subprocess
from tqdm import tqdm

DATASET_DIR = "dreambooth/dataset"
PROMPT_FILE = "dreambooth/dataset/prompts_and_classes.txt"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
OUTPUT_BASE_DIR = "lora-trained-dreambooth-xl"
MAX_TRAIN_STEPS = 500

def parse_prompts_and_classes(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    subject_to_class = {}
    parsing_classes = False
    header_skipped = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Classes" in line:
            parsing_classes = True
            continue
        if parsing_classes and not header_skipped:
            if line.lower() == "subject_name,class":
                header_skipped = True
                continue
        if parsing_classes and "," in line:
            subject, class_token = line.split(",")
            subject_to_class[subject.strip()] = class_token.strip()
        if "Prompts" in line:
            break
    return subject_to_class

def generate_prompt(subject_name, class_token):
    return f"a photo of sks {class_token}"

def train_instance(subject_name, class_token):
    instance_dir = os.path.join(DATASET_DIR, subject_name)
    output_dir = os.path.join(OUTPUT_BASE_DIR, subject_name)
    instance_prompt = generate_prompt(subject_name, class_token)
    validation_prompt = f"A photo of sks {class_token} in a bucket"

    command = [
        "accelerate", "launch", "train_dreambooth_lora_sdxl.py",
        "--pretrained_model_name_or_path", MODEL_NAME,
        "--instance_data_dir", instance_dir,
        "--pretrained_vae_model_name_or_path", VAE_PATH,
        "--output_dir", output_dir,
        "--mixed_precision", "fp16",
        "--instance_prompt", instance_prompt,
        "--resolution", "1024",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", "500",
        "--validation_prompt", validation_prompt,
        "--validation_epochs", "25",
        "--seed", "42"
    ]
    print(command)
    print(f"Training {subject_name} ({class_token})...")
    subprocess.run(command)

def main():
    subject_to_class = parse_prompts_and_classes(PROMPT_FILE)
    for subject_name, class_token in tqdm(subject_to_class.items(), total=len(subject_to_class), desc="Training instances"):
        train_instance(subject_name, class_token)

if __name__ == "__main__":
    main()
