from typing import List, Dict
import torch
import torch.nn.functional as F
import sys 
sys.path.append(".")
sys.path.append("..")


def cos_dist(a, b):
    """
        return 1 - cos(a, b)
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    res = a_norm @ b_norm.T

    return 1 - res

# def batch_gen_nn_map(
#     src_features: torch.Tensor,
#     src_masks: torch.Tensor,
#     tgt_features: torch.Tensor,
#     tgt_masks: torch.Tensor,
#     resolution: int,
#     device: torch.device = None,
#     batch_size: int = 64
# ) -> (torch.Tensor, torch.Tensor):
#     """
#     compute batch nearest neighbor mapping.

#     Args:
#         src_features: [B, C, H, W] source features
#         src_masks:    [B, H1, W1] source masks (bool or byte)
#         tgt_features: [B, C, H, W] target features
#         tgt_masks:    [B, H1, W1] target masks (not used, only keep interface consistent)
#         resolution:   int            the spatial resolution to which all features are interpolated (resolution×resolution)
#         device:       torch.device   if None, use the device of src_features
#         batch_size:   int            the batch size of target pixels when calculating distances

#     Returns:
#         nn_indices:   [B, resolution*resolution] every target pixel corresponds to the nearest source pixel index
#         nn_distances: [B, resolution*resolution] the corresponding minimum cosine distance
#     """
#     B, C, _, _ = src_features.shape
#     device = device or src_features.device
#     num_px = resolution * resolution

#     nn_indices = torch.zeros((B, num_px), dtype=torch.long,   device=device)
#     nn_dist   = torch.zeros((B, num_px), dtype=src_features.dtype, device=device)

#     for i in range(B):
#         sf = src_features[i]
#         sm = src_masks[i]
#         tf = tgt_features[i]

#         sf_r = F.interpolate(sf.unsqueeze(0), size=(resolution, resolution),
#                              mode='bilinear', align_corners=False).squeeze(0)  # [C, R, R]
#         tf_r = F.interpolate(tf.unsqueeze(0), size=(resolution, resolution),
#                              mode='bilinear', align_corners=False).squeeze(0)

#         src_flat = sf_r.view(C, -1).T  # [num_px, C]
#         tgt_flat = tf_r.view(C, -1).T

#         sm_r = F.interpolate(sm.unsqueeze(0).unsqueeze(0).float(),
#                              size=(resolution, resolution), mode='nearest'
#                             ).squeeze().bool()      # [R, R]
#         src_mask_flat = sm_r.view(-1)   # [num_px]

#         inds = torch.zeros(num_px, dtype=torch.long,   device=device)
#         dists = torch.zeros(num_px, dtype=src_features.dtype, device=device)

#         for j in range(0, num_px, batch_size):
#             end = j + batch_size
#             tgt_batch = tgt_flat[j:end]                    # [sub, C]
#             dist_mat  = cos_dist(src_flat, tgt_batch)      # [num_px, sub]
#             dist_mat[~src_mask_flat, :] = 2.0             
#             mn, idx = torch.min(dist_mat, dim=0)        
#             dists[j:end] = mn
#             inds [j:end] = idx

#         nn_indices[i] = inds
#         nn_dist   [i] = dists
        
#     return nn_indices, nn_dist

def batch_gen_nn_map(
    src_features: torch.Tensor,
    src_masks: torch.Tensor,
    tgt_features: torch.Tensor,
    tgt_masks: torch.Tensor,
    resolution: int,
    device: torch.device = None,
    batch_size: int = 64
) -> (torch.Tensor, torch.Tensor):
    """
    compute batch reverse nearest neighbor mapping:
    1. for each pixel in the source feature map with mask=1, find the nearest pixel in the target feature map
    2. reverse mapping, get the nearest source pixel index for each target pixel;
       for target pixels not mapped by any source pixel, randomly assign an index, set the distance to be greater than 1

    Returns:
        nn_indices: [B, R*R] the source pixel index corresponding to the target pixel
        nn_distances: [B, R*R] the corresponding cosine distance (unmapped points > 1)
    """
    B, C, _, _ = src_features.shape
    device = device or src_features.device
    num_px = resolution * resolution

    nn_indices = torch.zeros((B, num_px), dtype=torch.long, device=device)
    nn_distances = torch.zeros((B, num_px), dtype=src_features.dtype, device=device)

    for i in range(B):
        sf = src_features[i]
        sm = src_masks[i]
        tf = tgt_features[i]

        sf_r = F.interpolate(sf.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze(0)
        tf_r = F.interpolate(tf.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze(0)

        src_flat = sf_r.view(C, -1).T        # [R*R, C]
        tgt_flat = tf_r.view(C, -1).T        # [R*R, C]
        sm_r = F.interpolate(sm.unsqueeze(0).unsqueeze(0).float(), size=(resolution, resolution), mode='nearest').squeeze().bool()
        src_mask_flat = sm_r.view(-1)        # [R*R]

        src_indices = torch.nonzero(src_mask_flat, as_tuple=False).squeeze(1)
        M = src_indices.numel()

        inds = torch.randint(0, num_px, (num_px,), dtype=torch.long, device=device)
        dists = torch.ones((num_px,), dtype=src_features.dtype, device=device) * 1.1

        if M > 0:
            src2tgt_idx = torch.empty((M,), dtype=torch.long, device=device)
            src2tgt_dist = torch.empty((M,), dtype=src_features.dtype, device=device)

            for j in range(0, M, batch_size):
                end = j + batch_size
                batch_src_idx = src_indices[j:end]
                src_batch = src_flat[batch_src_idx]            # [sub, C]
                dist_mat = cos_dist(src_batch, tgt_flat)       # [sub, R*R]
                mn, idx = torch.min(dist_mat, dim=1)           
                src2tgt_dist[j:end] = mn
                src2tgt_idx[j:end] = idx

            # reverse: for each source->target mapping, update target->source, if the distance is smaller, overwrite
            for k in range(M):
                t = src2tgt_idx[k].item()
                s = src_indices[k].item()
                d = src2tgt_dist[k]
                if d < dists[t]:
                    dists[t] = d
                    inds[t] = s

        nn_indices[i] = inds
        nn_distances[i] = dists

    return nn_indices, nn_distances
