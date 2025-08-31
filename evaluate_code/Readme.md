# Evaluation Code Documentation

This directory contains evaluation scripts for assessing the quality of generated images using multiple metrics: CLIP-T (text-to-image), CLIP-I (image-to-image), and DINO (image-to-image similarity).

## 📊 Evaluation Metrics

### CLIP-T (Text-to-Image)
- **Description**: Measures the semantic alignment between text prompts and generated images
- **Weight**: We use a weight of **2.5** for CLIP-T, following the methodology described in ["CLIPScore: A Reference-free Evaluation Metric for Image Captioning"](https://arxiv.org/abs/2104.08718)
- **Formula**: `score = w * max(cosine_similarity(text_features, image_features), 0)`
- **Range**: Higher values indicate better text-image alignment

### CLIP-I (Image-to-Image)
- **Description**: Measures visual similarity between generated images and reference images using CLIP image features
- **Formula**: `score = cosine_similarity(generated_image_features, reference_image_features)`
- **Range**: Higher values indicate better visual similarity to reference images

### DINO (Image-to-Image)
- **Description**: Measures visual similarity using DINO (Self-Supervised Vision Transformer) features
- **Formula**: `score = cosine_similarity(dino_generated_features, dino_reference_features)`
- **Range**: Higher values indicate better visual similarity to reference images

## 📂 Files Description

### `evaluate_dreambooth.py`
- **Purpose**: Evaluates DreamBooth dataset results
- **Input Format**: Expects generated images with naming pattern `{name}_{*}_{prompt_tag}_0.png` in the output directory, where `prompt_tag` is derived from the first 40 characters of the prompt
- **Reference Images**: Looks for reference images in subdirectories under the dataset root directory (`{dataset_root}/{name}/`)
- **TSV Format**: Requires TSV file with `name` and `prompt` columns

### `evaluate_our_dataset.py`
- **Purpose**: Evaluates our held-out split dataset dataset results
- **Input Format**: Expects generated images with naming pattern `{index}_0.png` in the output directory
- **Reference Images**: Looks for reference images in the reference directory with `{index}.jpg` or `{index}.png`
- **TSV Format**: Requires TSV file with `index` and `prompt` columns


### ⚙️ Running DreamBooth Evaluation

```python
from evaluate_dreambooth import run_dreambooth_evaluation

# Example usage
clip_t, clip_i, dino = run_dreambooth_evaluation(
    tsv_path="/path/to/dreambooth_prompts.tsv",
    output_dir="/path/to/generated/images",
    dataset_root="/path/to/reference/images",
    clip_model_name="ViT-B/32",
    dino_model_name="dino_vits16",
    device="cuda",
    w=2.5,  # CLIP-T weight
    save_result=True
)

print(f"CLIP-T score: {clip_t:.4f}")
print(f"CLIP-I score: {clip_i:.4f}")
print(f"DINO score: {dino:.4f}")
```

### ⚙️ Running our held-out split dataset Evaluation

```python
from evaluate_our_dataset import evaluate_tsv
import clip
import torch

# Load models
device = 'cuda'
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).to(device)
dino_model.eval()

# Run evaluation
clip_t, clip_i, dino = evaluate_tsv(
    tsv_path="/path/to/prompts.tsv",
    output_dir="/path/to/generated/images",
    ref_dir="/path/to/reference/images",
    clip_model=clip_model,
    dino_model=dino_model,
    device=device,
    w=2.5  # CLIP-T weight
)

print(f"CLIP-T score: {clip_t:.4f}")
print(f"CLIP-I score: {clip_i:.4f}")
print(f"DINO score: {dino:.4f}")
```

### Command Line Usage

#### DreamBooth Evaluation
```bash
python evaluate_dreambooth.py
```
Modify the paths in the `__main__` section:
- `tsv_path`: Path to DreamBooth prompts TSV file
- `output_dir`: Directory containing generated images
- `dataset_root`: Directory containing reference images

#### Our held-out split Dataset Evaluation
```bash
python evaluate_our_dataset.py
```
Modify the paths in the `__main__` section:
- `tsv_path`: Path to prompts TSV file
- `output_dir`: Directory containing generated images
- `ref_dir`: Directory containing reference images

## Output

Both scripts will:
1. Print evaluation scores to console
2. Save results to `evaluation_result.txt` in the output directory
3. Return the three metric scores as floats
