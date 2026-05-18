import torch, torch.nn as nn
from mmcv.cnn import Linear
from mmengine import MODELS
from mmengine.model import BaseModule

import spconv.pytorch as spconv
from .utils import cartesian
from functools import partial


@MODELS.register_module()
class SparseConv3D(BaseModule):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        use_out_proj=True,
        kernel_size=5,
        dilation=1,
        init_cfg=None
    ):
        super().__init__(init_cfg)

        self.layer = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            dilation=dilation,
            indice_key='sub0')
        if use_out_proj:
            self.output_proj = Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()
        self.get_xyz = partial(cartesian, pc_range=pc_range)
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float), False)

    def forward(self, instance_feature, anchor, metas):
        # anchor: b, g, 11
        # instance_feature: b, g, c
        bs, g, _ = instance_feature.shape

        # vox_near = torch.tensor(metas[0]['vox_origin']).to(anchor.device)
        # scene_size = torch.tensor(metas[0]['scene_size']).to(anchor.device)
        # vox_near = metas[0]['vox_origin']
        # scene_size = metas[0]['scene_size']
        # vox_far = vox_near + scene_size
        # nyu_pc_range = torch.cat([vox_near, vox_far], dim=0).to(anchor.device)
        # nyu_pc_range = metas[0]['cam_vox_range'].to(anchor.device)
        nyu_pc_range = torch.stack([meta['cam_vox_range'] for meta in metas]).float()
        my_get = partial(cartesian, pc_range=nyu_pc_range)
        # sparsify
        # anchor_xyz = self.get_xyz(anchor).flatten(0, 1) 
        # import pdb; pdb.set_trace()
        anchor_xyz = my_get(anchor).flatten(0, 1) 
        indices = anchor_xyz - anchor_xyz.min(0, keepdim=True)[0]
        indices = indices / self.grid_size[None, :] # bg, 3
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
                bs, 1, 1).expand(-1, g, -1).flatten(0, 1),
            indices], dim=-1)

        spatial_shape = indices.max(0)[0]

        # print('spatial_shape', spatial_shape)

        # import pdb; pdb.set_trace()
        input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1), # bg, c [34560, 96]
            indices=batched_indices, # bg, 4 [34560, 4]
            spatial_shape=spatial_shape, # [58, 58, 35]
            batch_size=bs)

        output = self.layer(input)
        output = output.features.unflatten(0, (bs, g))

        return self.output_proj(output)
