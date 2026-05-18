# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Union
import torch, torch.nn as nn

from mmengine import MODELS
from mmengine.model import BaseModule
try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None
"""
anchor_encoder = dict(
    type='SparseGaussian3DEncoder',
    embed_dims=_dim_, 
    semantic_dim=cls_dims,
)
refine_layer = dict(
    type='SparseGaussian3DRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    restrict_xyz=True,
    unit_xyz=[4.0, 4.0, 1.0],
    refine_manual=[0, 1, 2],
    semantic_dim=cls_dims,
    semantics_activation=semantics_activation,
)
spconv_layer=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.8]*3,
    kernel_size=3,
)
spconv_layer_fillhead=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.8]*3,
    kernel_size=3,
    dilation=2
)
"""
@MODELS.register_module()
class SparseGaussianFormerOnline(BaseModule):
    def __init__(
        self,
        anchor_encoder,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        confidence_refine_layer: dict,
        frozen_refine_layer: dict,
        mid_refine_layer: dict = None,
        num_decoder: int = 6,
        spconv_layer: dict = None,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg):
            if cfg is None:
                return None
            return MODELS.build(cfg)
        
        self.anchor_encoder = build(anchor_encoder)
        self.op_config_map = {
            "norm": norm_layer,
            "ffn": ffn,
            "deformable": deformable_model,
            "refine": refine_layer,
            "confidence_refine": confidence_refine_layer,
            "frozen_refine": frozen_refine_layer,
            "mid_refine":mid_refine_layer,
            "spconv": spconv_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        anchor,
        instance_feature,
        feature_maps: Union[torch.Tensor, List], # mlvl_img_feats
        metas: dict,
        anchor_confidence
    ):
        
        if DAF is not None:
            feature_maps = DAF.feature_maps_format(feature_maps)

        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        anchor_embed = self.anchor_encoder(anchor) # [1, 21600, 96]

        prediction = []
        instance_feature_cache = instance_feature
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    metas)
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            elif op == "identity":
                identity = instance_feature
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            elif op == "deformable":
                # assert feature_queue is None and meta_queue is None and self.depth_module is None
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            # elif "refine" in op:
            elif op == "refine":
                anchor, gaussian, cls = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    metas,
                )
                prediction.append(anchor)
                
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            elif op == "confidence_refine":
                anchor, gaussian, cls = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    metas,
                    anchor_confidence,
                )
                prediction.append(anchor)
                
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            elif op == "frozen_refine":
                anchor, gaussian, cls = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    metas,
                    anchor_confidence,
                )
                prediction.append(anchor)
                
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if i == len(self.operation_order) - 1:
                    instance_feature_cache = instance_feature
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return prediction[-1], instance_feature_cache