import torch
from collections import defaultdict
from diffusers.utils.import_utils import is_xformers_available
from typing import Optional, List

class DIFTLatentStore:
    def __init__(self, steps: List[int], up_ft_indices: List[int]):
        self.steps = steps
        self.up_ft_indices = up_ft_indices
        self.dift_features = {}

    def __call__(self, features: torch.Tensor, t: int, layer_index: int):
        if t in self.steps and layer_index in self.up_ft_indices:
            self.dift_features[f'{int(t)}_{layer_index}'] = features

    def copy(self):
        copy_dift = DIFTLatentStore(self.steps, self.up_ft_indices)

        for key, value in self.dift_features.items():
            copy_dift.dift_features[key] = value.clone()

        return copy_dift

    def reset(self):
        self.dift_features = {}

