import torch.nn as nn, torch
from mmengine import MODELS
from mmengine.model import BaseModule
from .utils import linear_relu_ln


@MODELS.register_module()
class SparseGaussian3DEncoder(BaseModule):
    def __init__(
            self, 
            embed_dims: int = 256, # 96
            include_opa=True,
            semantic_dim=0, # 13
            include_v=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.include_opa = include_opa
        self.include_v = include_v

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.anchor_dim = 10 + int(include_opa) + semantic_dim + int(include_v) * 2
        self.encode_fc = embedding_layer(self.anchor_dim)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_3d: torch.Tensor):
        output = self.encode_fc(box_3d)
        output = self.output_fc(output)
        return output
