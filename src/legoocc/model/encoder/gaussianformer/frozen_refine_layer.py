from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Linear, Scale

from .utils import linear_relu_ln, safe_sigmoid, GaussianPrediction
import torch, torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class SparseGaussian3DDeltaForzenRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        restrict_xyz=False, # True
        unit_xyz=None, # [4.0, 4.0, 1.0]
        refine_manual=None, # [0, 1, 2]
        semantic_dim=0, # 13
        semantics_activation='softmax', # Identity
        include_opa=True,
        include_v=False
    ):
        super(SparseGaussian3DDeltaForzenRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = 10 + int(include_opa) + semantic_dim + int(include_v) * 2
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.restrict_xyz = restrict_xyz
        self.unit_xyz = unit_xyz # [4.0, 4.0, 1.0]
        if restrict_xyz:
            assert unit_xyz is not None
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            unit_sigmoid = [4 * unit_prob[i] for i in range(3)]
            self.unit_sigmoid = unit_sigmoid
        
        assert isinstance(refine_manual, list)
        self.refine_state = refine_manual
        assert all([self.refine_state[i] == i for i in range(len(self.refine_state))])

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        metas,
        anchor_confidence
    ):
        output = self.layers(instance_feature + anchor_embed) # 1, N, 23
        # import pdb; pdb.set_trace()
        anchor_updata_ratio = 1 - anchor_confidence
        anchor_updata_ratio[anchor_updata_ratio==0.5] = 0
        
        if self.restrict_xyz:
            delta_xyz_sigmoid = output[..., :3]
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1
            delta_xyz = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)
            output = torch.cat([delta_xyz, output[..., 3:]], dim=-1)
        
        if len(self.refine_state) > 0:
            refined_part_output = output[..., self.refine_state] * anchor_updata_ratio + anchor[..., self.refine_state]
            output = torch.cat([refined_part_output, output[..., len(self.refine_state):]], dim=-1)
            
        delta_scale = output[..., 3:6]
        scale_final = anchor[..., 3:6] + delta_scale * anchor_updata_ratio
        output = torch.cat([output[..., :3], scale_final, output[..., 6:]], dim=-1)

        rot = torch.nn.functional.normalize(output[..., 6:10], dim=-1)
        delta_w1, delta_x1, delta_y1, delta_z1 = rot[..., 0], rot[..., 1], rot[..., 2], rot[..., 3]
        # delta_w1[anchor_confidence.squeeze(-1)==1] = 1
        # delta_x1[anchor_confidence.squeeze(-1)==1] = 0
        # delta_y1[anchor_confidence.squeeze(-1)==1] = 0
        # delta_z1[anchor_confidence.squeeze(-1)==1] = 0
        delta_w1[anchor_confidence.squeeze(-1)>0.1] = 1
        delta_x1[anchor_confidence.squeeze(-1)>0.1] = 0
        delta_y1[anchor_confidence.squeeze(-1)>0.1] = 0
        delta_z1[anchor_confidence.squeeze(-1)>0.1] = 0
        
        w1, x1, y1, z1 = anchor[..., 6], anchor[..., 7], anchor[..., 8], anchor[..., 9]
        w_final = delta_w1 * w1 - delta_x1 * x1 - delta_y1 * y1 - delta_z1 * z1
        x_final = delta_w1 * x1 + delta_x1 * w1 + delta_y1 * z1 - delta_z1 * y1
        y_final = delta_w1 * y1 - delta_x1 * z1 + delta_y1 * w1 + delta_z1 * x1
        z_final = delta_w1 * z1 + delta_x1 * y1 - delta_y1 * x1 + delta_z1 * w1
        w_final = w_final.unsqueeze(-1)
        x_final = x_final.unsqueeze(-1)
        y_final = y_final.unsqueeze(-1)
        z_final = z_final.unsqueeze(-1)
        rot_final = torch.cat([w_final, x_final, y_final, z_final], dim=-1)
        rot_final = torch.nn.functional.normalize(rot_final, dim=-1)
        output = torch.cat([output[..., :6], rot_final, output[..., 10:]], dim=-1)
        
        delta_opa = output[..., 10: (10 + int(self.include_opa))]
        opa_final = anchor[..., 10: (10 + int(self.include_opa))] + delta_opa * anchor_updata_ratio
        output = torch.cat([output[..., :10], opa_final, output[..., (10 + int(self.include_opa)):]], dim=-1)
        
        delta_semantics = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        semantics_final = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)] + delta_semantics * anchor_updata_ratio
        output = torch.cat([output[..., :self.semantic_start], semantics_final, output[..., (self.semantic_start + self.semantic_dim):]], dim=-1)
        
        # del delta_scale, scale_final, delta_w1, delta_x1, delta_y1, delta_z1, w1, x1, y1, z1, w_final, x_final, y_final, z_final, rot_final, delta_opa, opa_final, delta_semantics, semantics_final
        
        nyu_pc_range = metas[0]['cam_vox_range'].to(output.device)
        xyz = safe_sigmoid(output[..., :3])
        xxx = xyz[..., 0] * (nyu_pc_range[3] - nyu_pc_range[0]) + nyu_pc_range[0]
        yyy = xyz[..., 1] * (nyu_pc_range[4] - nyu_pc_range[1]) + nyu_pc_range[1]
        zzz = xyz[..., 2] * (nyu_pc_range[5] - nyu_pc_range[2]) + nyu_pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        gs_scales = safe_sigmoid(output[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        opas = safe_sigmoid(output[..., 10: (10 + int(self.include_opa))])

        shs = torch.zeros(*instance_feature.shape[:-1], 0, 
            device=instance_feature.device, dtype=instance_feature.dtype)
        semantics = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opas,
            semantics=semantics
        )
        
        return output, gaussian, semantics