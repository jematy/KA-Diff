# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import batch_gen_nn_map
from math import sqrt
from skimage.filters import threshold_otsu
import numpy as np
class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        attn_layer = None,
        hidden_size=None,
        cross_attention_dim=None,
        ref_dift=None,
        tgt_dift=None,
        merged_layers=None,
        merged_times=None,
        ref_mask=None,
    ):
        super().__init__()
        self.already_run_times = 0
        self.attn_layer = attn_layer
        self.ref_dift = None
        self.tgt_dift = None
        self.merged_layers = None
        self.merged_times = None
        self.ref_mask = None
        self.tgt_mask = None

    def add_dift(
        self,
        ref_dift=None,
        tgt_dift=None,
        merged_layers=None,
        merged_times=None,
        ref_mask=None,
        tgt_mask=None,
        up_index=0,
    ):
        self.ref_dift = ref_dift
        self.tgt_dift = tgt_dift
        self.merged_layers = merged_layers
        self.merged_times = merged_times
        self.ref_mask = ref_mask
        self.tgt_mask = tgt_mask
        self.already_run_times = 0
        self.up_index = up_index
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        self.already_run_times += 1
        # if self.attn_layer == 3 and self.ref_dift is not None:
        #     # print(self.ref_dift["700_0"].shape)
        #     val = self.ref_dift["700_0"]
        #     print(f"[run {self.already_run_times}] shape={val.shape}  ptr={val.storage().data_ptr()}")
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        if self.merged_times is not None and self.already_run_times in self.merged_times and self.attn_layer in self.merged_layers:
            # print(self.attn_layer)
            # print(self.merged_layers)
            # print("[run {}] attn_layer={} merged_times={}".format(self.already_run_times, self.attn_layer, self.merged_times))
            query = query
            key = key
            value = value

            total_heads = query.shape[0] 
            D = query.shape[2]   
            batch_size = encoder_hidden_states.shape[0]
            heads_num = total_heads // batch_size
            gen_cond_slice = slice(heads_num*2, heads_num*3)
            ref_cond_slice = slice(heads_num*3, heads_num*4)

            q_gc = query[gen_cond_slice]   # [H, S, D]
            k_gc =   key[gen_cond_slice]   # [H, S, D]
            v_gc = value[gen_cond_slice]   # [H, S, D]

            q_rc = query[ref_cond_slice]   # [H, S, D]
            k_rc =   key[ref_cond_slice]   # [H, S, D]
            v_rc = value[ref_cond_slice]   # [H, S, D]

            resolution = int(sqrt(query.shape[1]))
            now_index = f"{20 * (51-self.already_run_times)}_{self.up_index}"
            # index_list = np.linspace(1000 - 0.0 * 1000 - 1, 0, 50).astype(np.int64)
            # step_ratio = 1000 * (1 - 0.1)// 50
            # index_list = (np.arange(0, 50) * step_ratio).round()[::-1].copy().astype(np.int64)
            # current_step_index = int(index_list[self.already_run_times-2])
            # now_index = f"{current_step_index}_{self.up_index}"
            ref_dift = self.ref_dift[now_index]
            tgt_dift = self.tgt_dift[now_index]
            _, ref_dift = ref_dift.chunk(2)
            _, _, tgt_dift, _ = tgt_dift.chunk(4)
            nn_indices, nn_dist = batch_gen_nn_map(
                src_features = ref_dift,
                src_masks = self.ref_mask,
                tgt_features = tgt_dift,
                tgt_masks = None,
                resolution = resolution,
                device = tgt_dift.device,
            )

            H, S, D = q_gc.shape
            nn = nn_indices.squeeze(0)
            nn = nn.unsqueeze(0).expand(heads_num, -1)
            nn_idx_3d = nn.unsqueeze(-1).expand(-1, -1, D)
            threshold = 0.7
            dist = nn_dist.squeeze(0)  
            dist_np = dist.detach().cpu().numpy()
            # threshold = threshold_otsu(dist_np)
            S = dist.shape[0]
            mask_seq = (dist < threshold)
            q_replaced = torch.gather(q_rc, dim=1, index=nn_idx_3d)  # [H, S, D]
            # only replace the positions where mask_seq is True
            mask3d = mask_seq.unsqueeze(-1).expand_as(q_gc)          # [H, S, D]
            q_gc = torch.where(mask3d, q_replaced, q_gc)

            # same for K, V:
            k_replaced = torch.gather(k_rc, dim=1, index=nn_idx_3d)
            k_gc       = torch.where(mask3d, k_replaced, k_gc)

            v_replaced = torch.gather(v_rc, dim=1, index=nn_idx_3d)
            v_gc       = torch.where(mask3d, v_replaced, v_gc)

            # finally, put q_gc, k_gc, v_gc back into the large tensor of query/key/value, continue the subsequent process:
            # query[gen_cond_slice] = q_gc
            key[gen_cond_slice]   = k_gc# * 0.8 + key[gen_cond_slice] * 0.2
            value[gen_cond_slice] = v_gc# * 0.8 + value[gen_cond_slice] * 0.2
            # then calculate attention according to the original process:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states   = torch.bmm(attention_probs, value)
            hidden_states   = attn.batch_to_head_dim(hidden_states)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4,
        attn_layer = None,
        ref_dift=None,
        tgt_dift=None,
        merged_layers=None,
        merged_times=None,
        ref_mask=None,):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.already_run_times = 0
        self.attn_layer = attn_layer
        self.ref_dift = None
        self.tgt_dift = None
        self.merged_layers = None
        self.merged_times = None
        self.ref_mask = None
        self.tgt_mask = None

    def add_dift(
        self,
        ref_dift=None,
        tgt_dift=None,
        merged_layers=None,
        merged_times=None,
        ref_mask=None,
        tgt_mask=None,
        up_index=0,
    ):
        self.ref_dift = ref_dift
        self.tgt_dift = tgt_dift
        self.merged_layers = merged_layers
        self.merged_times = merged_times
        self.ref_mask = ref_mask
        self.tgt_mask = tgt_mask
        self.already_run_times = 0
        self.up_index = up_index
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # if self.merged_times is not None and self.already_run_times in self.merged_times and self.attn_layer in self.merged_layers:
        #     # print("[run {}] attn_layer={} merged_times={}".format(self.already_run_times, self.attn_layer, self.merged_times))
        #     query = query
        #     key = key
        #     value = value

        #     total_heads = query.shape[0] 
        #     D = query.shape[2]   
        #     batch_size = encoder_hidden_states.shape[0]
        #     heads_num = total_heads // batch_size
        #     gen_cond_slice = slice(heads_num*2, heads_num*3)
        #     ref_cond_slice = slice(heads_num*3, heads_num*4)

        #     q_gc = query[gen_cond_slice]   # [H, S, D]
        #     k_gc =   key[gen_cond_slice]   # [H, S, D]
        #     v_gc = value[gen_cond_slice]   # [H, S, D]

        #     q_rc = query[ref_cond_slice]   # [H, S, D]
        #     k_rc =   key[ref_cond_slice]   # [H, S, D]
        #     v_rc = value[ref_cond_slice]   # [H, S, D]

        #     resolution = int(sqrt(query.shape[1]))
        #     now_index = f"{20 * (51-self.already_run_times)}_0"
        #     ref_dift = self.ref_dift[now_index]
        #     tgt_dift = self.tgt_dift[now_index]
        #     _, ref_dift = ref_dift.chunk(2)
        #     _, tgt_dift = tgt_dift.chunk(2)
        #     nn_indices, nn_dist = batch_gen_nn_map(
        #         src_features = ref_dift,
        #         src_masks = self.ref_mask,
        #         tgt_features = tgt_dift,
        #         tgt_masks = None,
        #         resolution = resolution,
        #         device = tgt_dift.device,
        #     )

        #     H, S, D = q_gc.shape
        #     nn = nn_indices.squeeze(0)
        #     nn = nn.unsqueeze(0).expand(heads_num, -1)
        #     nn_idx_3d = nn.unsqueeze(-1).expand(-1, -1, D)
        #     threshold = 0.7
        #     dist = nn_dist.squeeze(0)  
        #     S = dist.shape[0]
        #     mask_seq = (dist < threshold)
        #     q_replaced = torch.gather(q_rc, dim=1, index=nn_idx_3d)  # [H, S, D]
        #     mask3d = mask_seq.unsqueeze(-1).expand_as(q_gc)          # [H, S, D]
        #     q_gc = torch.where(mask3d, q_replaced, q_gc)

        #     k_replaced = torch.gather(k_rc, dim=1, index=nn_idx_3d)
        #     k_gc       = torch.where(mask3d, k_replaced, k_gc)

        #     v_replaced = torch.gather(v_rc, dim=1, index=nn_idx_3d)
        #     v_gc       = torch.where(mask3d, v_replaced, v_gc)

        #     query[gen_cond_slice] = q_gc
        #     key[gen_cond_slice]   = k_gc
        #     value[gen_cond_slice] = v_gc
        #     attention_probs = attn.get_attention_scores(query, key, attention_mask)
        #     hidden_states   = torch.bmm(attention_probs, value)
        #     hidden_states   = attn.batch_to_head_dim(hidden_states)
        # else:
        #     attention_probs = attn.get_attention_scores(query, key, attention_mask)
        #     hidden_states = torch.bmm(attention_probs, value)
        #     hidden_states = attn.batch_to_head_dim(hidden_states)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states