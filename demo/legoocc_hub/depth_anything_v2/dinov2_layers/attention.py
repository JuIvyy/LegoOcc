# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    # def forward(self, x: Tensor) -> Tensor:
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    #     q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    #     attn = q @ k.transpose(-2, -1)

    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x
    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        """
        x: (B, N, C)
        attn_mask: 可选，广播到 (B, num_heads, N, N) 的布尔或加性掩码
        is_causal: 若为 True，启用下三角掩码（decoder 类自回归场景）
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 形状均为 (B, num_heads, N, head_dim)

        # ⚠️ 不需要手动缩放，SDPA 内部会除以 sqrt(head_dim)
        # dropout 仅在训练时生效
        attn_dropout_p = self.attn_drop.p if self.training and self.attn_drop.p > 0 else 0.0

        # PyTorch SDPA：会自动选择 Flash / mem-efficient / math 实现
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,          # None 或 布尔/加性 mask（形状可广播）
            dropout_p=attn_dropout_p,
            is_causal=is_causal
        )  # (B, num_heads, N, head_dim)

        out = out.transpose(1, 2).reshape(B, N, C)  # -> (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        