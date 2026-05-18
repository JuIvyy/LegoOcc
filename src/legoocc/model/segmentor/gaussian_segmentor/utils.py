import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

import torch

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def inverse_sigmoid(x):
    return np.log(x/((1-x)+1e-10))


def _markley_group_mean(qA, qB_nei, w_nei=None):
    """
    qA:        [4]          (this A's quaternion)
    qB_nei:    [K,4]        (neighbors' quaternions for this A)
    w_nei:     [K] or None  (weights per neighbor; if None -> ones)
    returns:   q_avg [4], unit quaternion; sign aligned with qA
    """
    if qB_nei.numel() == 0:
        # no neighbors -> fallback to qA
        qs = qA / qA.norm()
        return qs
    if w_nei is None:
        w_nei = torch.ones(qB_nei.size(0), device=qB_nei.device, dtype=qB_nei.dtype)
    # build 4x4 symmetric M = wA*qA qA^T + sum_k w_k*q_k q_k^T
    # A's own weight (can be a knob; keep 1.0 by default)
    wA = 1.0
    qA = qA / (qA.norm() + 1e-12)
    M = wA * torch.ger(qA, qA)  # [4,4]
    # neighbors
    # out_k = w_k * q_k q_k^T -> scatter-sum (here we just sum in a loop for clarity; K is small per A)
    # If you prefer vectorized: use einsum('ki,kj,k->kij', qB_nei, qB_nei, w_nei).sum(0)
    Mb = torch.einsum('ki,kj,k->ij', qB_nei, qB_nei, w_nei)  # [4,4]
    M = M + Mb
    # principal eigenvector of M
    evals, evecs = torch.linalg.eigh(M)   # ascending
    q = evecs[:, -1]
    # align sign to qA to avoid jumps
    if (q * qA).sum() < 0:
        q = -q
    q = q / (q.norm() + 1e-12)
    return q



DTYPE_INTERMEDIATE = torch.float32

def _to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return x if isinstance(x, tuple) else (int(x), int(x))


# https://github.com/rwightman/timm/blob/main/timm/layers/patch_embed.py#L302
def resample_patch_embed(
        patch_embed: torch.Tensor,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """ Standalone function (computes matrix on each call). """
    assert len(patch_embed.shape) == 4, "Input tensor should be 4D (out_ch, in_ch, h, w)"
    assert len(new_size) == 2, "New shape should only be hw (height, width)"

    old_size_tuple: Tuple[int, int] = tuple(patch_embed.shape[-2:])
    new_size_tuple: Tuple[int, int] = tuple(new_size)

    if old_size_tuple == new_size_tuple:
        return patch_embed

    device = patch_embed.device
    orig_dtype = patch_embed.dtype

    resize_mat = _compute_resize_matrix(
        old_size_tuple, new_size_tuple, interpolation, antialias, device, DTYPE_INTERMEDIATE
    )
    pinv_matrix = torch.linalg.pinv(resize_mat)  # Calculates the pseudoinverse matrix used for resampling
    resampled_patch_embed = _apply_resampling(
        patch_embed, pinv_matrix, new_size_tuple, orig_dtype, DTYPE_INTERMEDIATE
    )
    return resampled_patch_embed


def _apply_resampling(
    patch_embed: torch.Tensor,
    pinv_matrix: torch.Tensor,
    new_size_tuple: Tuple[int, int],
    orig_dtype: torch.dtype,
    intermediate_dtype: torch.dtype = DTYPE_INTERMEDIATE
) -> torch.Tensor:
    """ Simplified resampling w/o vmap use.
    As proposed by https://github.com/stas-sl
    """
    c_out, c_in, *_ = patch_embed.shape
    patch_embed = patch_embed.reshape(c_out, c_in, -1).to(dtype=intermediate_dtype)
    pinv_matrix = pinv_matrix.to(dtype=intermediate_dtype)
    resampled_patch_embed = patch_embed @ pinv_matrix  # (C_out, C_in, P_old * P_old) @ (P_old * P_old, P_new * P_new)
    resampled_patch_embed = resampled_patch_embed.reshape(c_out, c_in, *new_size_tuple).to(dtype=orig_dtype)
    return resampled_patch_embed


def _compute_resize_matrix(
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
    interpolation: str,
    antialias: bool,
    device: torch.device,
    dtype: torch.dtype = DTYPE_INTERMEDIATE
) -> torch.Tensor:
    """Computes the resize matrix basis vectors and interpolates them to new_size."""
    old_h, old_w = old_size
    new_h, new_w = new_size
    old_total = old_h * old_w
    new_total = new_h * new_w

    eye_matrix = torch.eye(old_total, device=device, dtype=dtype)
    basis_vectors_batch = eye_matrix.reshape(old_total, 1, old_h, old_w)
    resized_basis_vectors_batch = F.interpolate(
        basis_vectors_batch,
        size=new_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False
    ) # Output shape: (old_total, 1, new_h, new_w)
    resize_matrix = resized_basis_vectors_batch.squeeze(1).permute(1, 2, 0).reshape(new_total, old_total)
    return resize_matrix # Shape: (new_total, old_total)


@torch.no_grad()
def _interp_pos_embed_grid_only(pos_embed: torch.Tensor, num_prefix_tokens: int,
                                new_hw: Tuple[int,int], antialias: bool=True) -> torch.Tensor:
    """只插值网格部分；CLS 前缀保持不变。pos_embed: [1, 1+Gh*Gw, C]"""
    B, N, C = pos_embed.shape
    assert B == 1 and num_prefix_tokens == 1, "本函数假设只有 CLS 前缀。"
    grid = pos_embed[:, 1:, :]                                # [1, Gh*Gw, C]
    gh0 = int(round(math.sqrt(grid.shape[1]))); gw0 = grid.shape[1] // gh0
    grid = grid.permute(0,2,1).reshape(1, C, gh0, gw0)        # [1,C,gh0,gw0]
    grid = F.interpolate(grid, size=new_hw, mode="bicubic", align_corners=False, antialias=antialias)
    grid = grid.flatten(2).permute(0,2,1)                     # [1, Gh1*Gw1, C]
    return torch.cat([pos_embed[:, :1, :], grid], dim=1)


@torch.no_grad()
def set_input_size(
    model,  # DinoVisionTransformer（无 register_tokens）
    img_size: Optional[Union[int, Tuple[int, int]]] = None,
    patch_size: Optional[Union[int, Tuple[int, int]]] = None,
):
    """
    仅调整 patch embedding & 相关尺寸记录；pos_embed 不处理（模型内部会 interpolate_pos_encoding）。
    """
    assert hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"), "model.patch_embed.proj 缺失"

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    # 1) 解析目标尺寸
    cur_img = getattr(model.patch_embed, "img_size", None)
    if img_size is None:
        if cur_img is None:
            raise ValueError("请显式传入 img_size（model.patch_embed 没有记录 img_size）")
        tgt_img = _to_2tuple(cur_img)
    else:
        tgt_img = _to_2tuple(img_size)

    kH, kW = model.patch_embed.proj.kernel_size
    cur_ps = (kH, kW)
    tgt_ps = _to_2tuple(patch_size if patch_size is not None else kH)
    if tgt_ps[0] != tgt_ps[1]:
        raise ValueError("目前仅支持方形 patch_size")

    # 2) 如 patch_size 改变：重建 proj + 重采样权重
    if cur_ps != tgt_ps:
        old_w = model.patch_embed.proj.weight.detach().clone()
        old_b = None if model.patch_embed.proj.bias is None else model.patch_embed.proj.bias.detach().clone()

        new_proj = nn.Conv2d(
            in_channels  = model.patch_embed.proj.in_channels,
            out_channels = model.patch_embed.proj.out_channels,
            kernel_size  = tgt_ps,
            stride       = tgt_ps,
            bias         = (old_b is not None),
        ).to(device=device, dtype=dtype)

        # 用你提供的 timm 风格重采样，把 4D 卷积核插值到新 kernel 尺寸
        new_w = resample_patch_embed(old_w, list(tgt_ps), interpolation='bicubic', antialias=True, verbose=False)
        new_proj.weight.copy_(new_w.to(dtype))
        if old_b is not None:
            new_proj.bias.copy_(old_b.to(dtype))
        model.patch_embed.proj = new_proj

        #（可选）同步 model.patch_size 字段
        if hasattr(model, "patch_size"):
            model.patch_size = tgt_ps[0]

    # 3) 更新 PatchEmbed 的尺寸记录
    if hasattr(model.patch_embed, "_init_img_size"):
        model.patch_embed.img_size, model.patch_embed.grid_size, model.patch_embed.num_patches = \
            model.patch_embed._init_img_size(tgt_img)
    else:
        Gh = math.ceil(tgt_img[0] / tgt_ps[0])
        Gw = math.ceil(tgt_img[1] / tgt_ps[1])
        model.patch_embed.img_size  = tgt_img
        model.patch_embed.patches_resolution = (Gh, Gw)
        model.patch_embed.num_patches = Gh * Gw
        model.patch_embed.patch_size = tgt_ps

    # 不处理 pos_embed：模型 forward 时自行 interpolate_pos_encoding
    return model


class DeformAttn2D(nn.Module):
    """
    单尺度 Deformable Attention:
      - Query: feat (B, Nq, C)
      - Ref points: xyz (B, Nq, 2), 默认 [0,1] 归一化到 W/H 方向
      - Key/Value: fm_feats (B, C, H, W)
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = d_model // n_heads

        # 线性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)

        # 由 query 预测 offsets 和 attention weights
        # offsets: (dx, dy) * (n_heads * n_points)
        self.offsets = nn.Linear(d_model, n_heads * n_points * 2)
        # 注意力权重： (n_heads * n_points)
        self.attn_w = nn.Linear(d_model, n_heads * n_points)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        # 初始化（参考 Deformable-DETR 的做法：偏移初始化为 0）
        nn.init.zeros_(self.offsets.weight); nn.init.zeros_(self.offsets.bias)
        nn.init.zeros_(self.attn_w.weight);  nn.init.zeros_(self.attn_w.bias)

    @staticmethod
    def _to_grid(coords01: torch.Tensor):
        """
        将 [0,1] 的 (x,y) 归一化坐标转成 grid_sample 需要的 [-1,1]。
        coords01: (B, *, 2)  [x_in_0_1, y_in_0_1]
        return:  (B, *, 2)   [x_in_-1_1, y_in_-1_1]
        """
        return coords01 * 2.0 - 1.0

    def forward(self, feat: torch.Tensor, xyz: torch.Tensor, fm_feats: torch.Tensor):
        """
        feat:     (B, Nq, C)
        xyz:      (B, Nq, 2)  参考点（默认 [0,1]，分别对应 x/w 与 y/h）
        fm_feats: (B, C, H, W)
        return:   (B, Nq, C)  更新后的 query
        """
        B, Nq, C = feat.shape
        _, Ck, H, W = fm_feats.shape
        assert Ck == C, "fm_feats channel must equal d_model"

        # 1) 线性投影
        q = self.q_proj(feat)                                 # (B, Nq, C)
        v_map = self.v_proj(fm_feats)                         # (B, C, H, W)

        # 2) 由 query 预测 offsets & attention weights
        #    offsets: (B, Nq, n_heads, n_points, 2)  —— 相对参考点的小偏移（在 [0,1] 语义下）
        offsets = self.offsets(q).view(B, Nq, self.n_heads, self.n_points, 2)
        #    attn:    (B, n_heads, Nq, n_points)
        attn = self.attn_w(q).view(B, Nq, self.n_heads, self.n_points).permute(0, 2, 1, 3)
        attn = F.softmax(attn, dim=-1)

        # 3) 构造采样网格（grid_sample 需要 [-1,1]）
        #    参考点 xyz: (B, Nq, 2) in [0,1]，偏移 offsets 也视为 [0,1] 空间的相对偏移
        ref = xyz.unsqueeze(2).unsqueeze(3)                   # (B, Nq, 1, 1, 2)
        pts01 = ref + offsets                                 # (B, Nq, n_heads, n_points, 2)
        #    夹紧到 [0,1]
        pts01 = pts01.clamp(0.0, 1.0)
        #    转成 grid_space [-1,1]，并整理维度以适配 grid_sample
        #    目标形状： (B*n_heads, Nq*n_points, 1, 2)
        pts_grid = self._to_grid(pts01).permute(0, 2, 1, 3, 4).contiguous()    # (B, n_heads, Nq, n_points, 2)
        Bn = B * self.n_heads
        grid = pts_grid.view(Bn, Nq * self.n_points, 1, 2)

        # 4) 按 head 切分 value（B*n_heads, d_head, H, W）并采样
        v = v_map.view(B, self.n_heads, self.d_head, H, W).contiguous()
        v = v.view(Bn, self.d_head, H, W)                     # (B*n_heads, d_head, H, W)

        # 双线性采样：输出 (B*n_heads, d_head, Nq*n_points, 1)
        sampled = F.grid_sample(
            v, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled = sampled.view(B, self.n_heads, self.d_head, Nq, self.n_points)  # (B, H, Dh, Nq, P)

        # 5) 加权聚合（对 n_points 做加权求和）
        # attn: (B, n_heads, Nq, n_points) -> (B, n_heads, 1, Nq, n_points)
        attn_w = attn.unsqueeze(2)
        out = (sampled * attn_w).sum(dim=-1)                  # (B, n_heads, d_head, Nq)
        out = out.permute(0, 3, 1, 2).contiguous().view(B, Nq, C)  # (B, Nq, C)

        # 6) 输出线性层
        out = self.out_proj(out)                              # (B, Nq, C)
        return out