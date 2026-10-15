"""Pure-PyTorch multi-scale deformable attention.

Same parameters and forward semantics as `mmcv.ops.MultiScaleDeformableAttention`
(batch_first=True, 2-d reference points), so checkpoints load unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)
    output = output.view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, value, query_pos, reference_points, spatial_shapes):
        """
        query: (B, Q, C); value: (B, sum(H*W), C); query_pos: (B, Q, C)
        reference_points: (B, Q, L, 2) normalized (x, y); spatial_shapes: (L, 2) as (h, w)
        """
        identity = query
        query = query + query_pos
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        value = self.value_proj(value).view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return self.dropout(output) + identity
