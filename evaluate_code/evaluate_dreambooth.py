import pandas as pd
import os
# os.environ["HF_DIFFUSERS_CACHE"] = "/root/autodl-tmp"
# os.environ["no_proxy"] = "localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn"
# os.environ["http_proxy"] = "http://172.20.0.113:12798"
# os.environ["https_proxy"] = "http://172.20.0.113:12798"
# os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
# os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
import argparse
import glob
import json

import warnings
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from packaging import version
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

# New added
import pickle

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, append=False, prefix='A photo depicts'):
        self.data = data
        self.prefix = ''
        if append:
            self.prefix = prefix
            if self.prefix[-1] != ' ':
                self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


def Convert(image):
    return image.convert("RGB")


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8, append=False):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, append=append),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        # for b in tqdm(data):
        for b in data:
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, datasetclass, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        datasetclass(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        # for b in tqdm(data):
        for b in data:
            b = b['image'].to(device)
            if hasattr(model, 'encode_image'):
                if device == 'cuda':
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
            else:
                all_image_features.append(model(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, append=False, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device, append=append)

    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / \
            np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per


def clipeval(image_dir, candidates_json, model, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG')) and 'gen' in path]
    image_ids = [Path(path).stem + '.png' for path in image_paths]
    with open(candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]

    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)

    _, per_instance_image_text = get_clip_score(
        model, image_feats, candidates, device)

    scores = {image_id: {'CLIPScore': float(clipscore)}
              for image_id, clipscore in
              zip(image_ids, per_instance_image_text)}

    return np.mean([s['CLIPScore'] for s in scores.values()]), np.std([s['CLIPScore'] for s in scores.values()])


def clipeval_image(image_dir, image_dir_ref, model, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG')) and 'gen' in path]
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)
    image_feats_ref = extract_all_images(
        image_paths_ref, model, CLIPImageDataset, device, batch_size=64, num_workers=8)
    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)


def dinoeval_image(image_dir, image_dir_ref, model, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG')) and 'gen' in path]
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    image_feats = extract_all_images(
        image_paths, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)

import re
def evaluate_dreambooth_tsv(tsv_path, output_dir, dataset_root, clip_model, dino_model, device, w=2.5):
    df = pd.read_csv(tsv_path, sep='\t')
    
    valid_prompts = []
    gen_paths = []
    ref_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        name = row['name']
        prompt = row['prompt']

        subject_path = os.path.join(dataset_root, name)
        if not os.path.isdir(subject_path):
            warnings.warn(f"[skip] Reference image directory not found: {subject_path}")
            continue

        ref_candidates = [
            f for f in os.listdir(subject_path)
            if f.lower().endswith(('.jpg', '.png')) and not f.lower().endswith('_mask.jpg')
        ]
        if not ref_candidates:
            warnings.warn(f"[skip] No reference image found: {subject_path}")
            continue

        ref_img_path = os.path.join(subject_path, ref_candidates[0])

        prompt_tag = re.sub(r'[^\w\s]', '', prompt[:40]).replace(' ', '_')
        pattern = f"{name}_*_({prompt_tag})_0.png"
        gen_img_candidates = glob.glob(os.path.join(output_dir, f"{name}_*_{prompt_tag}_0.png"))

        if not gen_img_candidates:
            warnings.warn(f"[skip] No generated image found: {name} - prompt={prompt}")
            continue

        gen_img_path = gen_img_candidates[0]

        valid_prompts.append(prompt)
        gen_paths.append(gen_img_path)
        ref_paths.append(ref_img_path)

    if len(gen_paths) == 0:
        raise RuntimeError("No valid (gen_path + ref_path) pairs found, cannot evaluate")

    # ---- CLIP-Text → Image ----
    text_feats = extract_all_captions(valid_prompts, clip_model, device, append=False)
    image_feats = extract_all_images(gen_paths, clip_model, CLIPImageDataset, device)
    text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)
    image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
    clip_t_per = w * np.clip(np.sum(text_feats * image_feats, axis=1), 0, None)
    clip_t_mean = float(np.mean(clip_t_per))

    # ---- CLIP-Image → Image ----
    ref_feats = extract_all_images(ref_paths, clip_model, CLIPImageDataset, device)
    ref_feats = ref_feats / np.linalg.norm(ref_feats, axis=1, keepdims=True)
    clip_i_per = np.sum(image_feats * ref_feats, axis=1)
    clip_i_mean = float(np.mean(clip_i_per))

    # ---- DINO-Image → Image ----
    dino_gen = extract_all_images(gen_paths, dino_model, DINOImageDataset, device)
    dino_ref = extract_all_images(ref_paths, dino_model, DINOImageDataset, device)
    dino_gen = dino_gen / np.linalg.norm(dino_gen, axis=1, keepdims=True)
    dino_ref = dino_ref / np.linalg.norm(dino_ref, axis=1, keepdims=True)
    dino_per = np.sum(dino_gen * dino_ref, axis=1)
    dino_mean = float(np.mean(dino_per))

    return clip_t_mean, clip_i_mean, dino_mean

def run_dreambooth_evaluation(tsv_path, output_dir, dataset_root,
                              clip_model_name="ViT-B/32", dino_model_name="dino_vits16",
                              device="cuda", w=2.5, save_result=True):
    """
    Run DreamBooth evaluation, return CLIP-T, CLIP-I and DINO scores.

    Parameters:
    - tsv_path: TSV file path, contains name and prompt columns.
    - output_dir: Generated image directory.
    - dataset_root: Dataset root directory (reference image存放位置).
    - clip_model_name: CLIP model name (default "ViT-B/32").
    - dino_model_name: DINO model name (default "dino_vits16").
    - device: "cuda" or "cpu".
    - w: Weight parameter in CLIP-T.
    - save_result: Whether to save results to output_dir.

    Returns:
    - clip_t_mean, clip_i_mean, dino_mean
    """
    import clip
    import torch
    import os

    print("Load CLIP model...")
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model.eval()

    print("Load DINO model...")
    dino_model = torch.hub.load('facebookresearch/dino:main', dino_model_name, pretrained=True).to(device)
    dino_model.eval()

    print("Start evaluating...")
    clip_t, clip_i, dino = evaluate_dreambooth_tsv(tsv_path, output_dir, dataset_root,
                                                   clip_model, dino_model, device, w=w)

    if save_result:
        result_txt_path = os.path.join(output_dir, "evaluation_result.txt")
        with open(result_txt_path, "w") as f:
            f.write(f"CLIP-T (text→image) score: {clip_t:.4f}\n")
            f.write(f"CLIP-I (image→image) score: {clip_i:.4f}\n")
            f.write(f"DINO    (image→image) score: {dino:.4f}\n")
        print(f"Result txt: {result_txt_path}")

    return clip_t, clip_i, dino

if __name__ == '__main__':
    device = 'cuda'
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).to(device)
    dino_model.eval()

    tsv_path = '/root/autodl-tmp/mip-adapter_finegrain/dreambooth_prompts.tsv'
    output_dir = '/root/autodl-tmp/mip-adapter_finegrain/ablation_final/dreambooth_dataset_latent_seed42_XLplus_seed42_scale5_threshold_9'
    dataset_root = '/root/autodl-tmp/dreambooth_single_image/dataset'

    clip_t, clip_i, dino = evaluate_dreambooth_tsv(tsv_path, output_dir, dataset_root,
                                                   clip_model, dino_model, device)

    print(f"CLIP-T (text→image) score: {clip_t:.4f}")
    print(f"CLIP-I (image→image) score: {clip_i:.4f}")
    print(f"DINO    (image→image) score: {dino:.4f}")

    result_txt_path = os.path.join(output_dir, "evaluation_result.txt")
    with open(result_txt_path, "w") as f:
        f.write(f"CLIP-T (text→image) score: {clip_t:.4f}\n")
        f.write(f"CLIP-I (image→image) score: {clip_i:.4f}\n")
        f.write(f"DINO    (image→image) score: {dino:.4f}\n")

    print(f"Result txt: {result_txt_path}")