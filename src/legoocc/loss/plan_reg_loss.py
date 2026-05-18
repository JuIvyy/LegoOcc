from .base_loss import BaseLoss
from . import GPD_LOSS
import torch
import numpy as np

@GPD_LOSS.register_module()
class PlanRegLoss(BaseLoss):
    def __init__(self, weight=1.0, num_modes=3, input_dict=None, loss_type='l2', loss_name=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'rel_pose': 'rel_pose',
                'gt_rel_pose': 'gt_rel_pose',
                'gt_pose_mode': 'gt_pose_mode',
            }
        else:
            self.input_dict = input_dict
        if loss_name is not None:
            self.loss_name = loss_name
        self.loss_func = self.plan_reg_loss
        self.num_mode = num_modes
        self.loss_type = loss_type
        assert loss_type in ['l1', 'l2'], f'loss_type {loss_type} not supported'
        
    def plan_reg_loss(self, rel_pose, gt_rel_pose, gt_pose_mode):
        rel_pose = rel_pose.float()
        bs, num_frames, num_modes, _ = rel_pose.shape
        rel_pose = rel_pose.transpose(1, 2) # B, M, F, 2
        gt_pose_mode = gt_pose_mode.transpose(1,2) # B, F, M -> B, M, F
        gt_rel_pose = gt_rel_pose.unsqueeze(1).repeat(1, num_modes, 1, 1) # B, M, F, 2
            
        if self.loss_type == 'l1':
            weight = gt_pose_mode[..., None].repeat(1, 1, 1, 2)
            loss = torch.abs(rel_pose - gt_rel_pose) * weight
        elif self.loss_type == 'l2':
            weight = gt_pose_mode
            loss = torch.sqrt(((rel_pose - gt_rel_pose) ** 2).sum(-1)) * weight
        
        return loss.sum() / bs / num_frames


@GPD_LOSS.register_module()
class PlanRotLoss(BaseLoss):
    def __init__(self, weight=1.0, num_modes=3, input_dict=None, loss_type='l2', loss_name=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'rel_rot': 'rel_rot',
                'gt_rel_rot': 'gt_rel_rot',
                'gt_pose_mode': 'gt_pose_mode',
            }
        else:
            self.input_dict = input_dict
        if loss_name is not None:
            self.loss_name = loss_name
        self.loss_func = self.plan_rot_loss
        self.num_mode = num_modes
        self.loss_type = loss_type
        assert loss_type in ['l1', 'l2'], f'loss_type {loss_type} not supported'
        
    def plan_rot_loss(self, rel_rot, gt_rel_rot, gt_pose_mode):
        rel_rot = rel_rot.float()
        bs, num_frames, num_modes = rel_rot.shape
        gt_rel_rot = gt_rel_rot.unsqueeze(2).repeat(1, 1, num_modes)
            
        if self.loss_type == 'l1':
            weight = gt_pose_mode
            loss = torch.abs(rel_rot - gt_rel_rot) * weight #* 180 / torch.pi
        elif self.loss_type == 'l2':
            weight = gt_pose_mode
            loss = ((rel_rot - gt_rel_rot) ** 2) * weight #* 180 / torch.pi
        
        return loss.sum() / bs / num_frames


@GPD_LOSS.register_module()
class PlanL2Loss(BaseLoss):
    def __init__(self, weight=1.0, num_modes=3, input_dict=None, loss_type='l2', **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'rel_pose': 'rel_pose',
                'rel_rot': 'rel_rot',
                'gt_rel_pose': 'gt_rel_pose',
                'gt_pose_mode': 'gt_pose_mode',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.plan_l2_loss
        self.num_mode = num_modes
        self.loss_type = loss_type
        assert loss_type in ['l1', 'l2'], f'loss_type {loss_type} not supported'
        
    def plan_l2_loss(self, rel_pose, rel_rot, gt_rel_pose, gt_pose_mode):
        rel_pose = rel_pose.float()
        rel_rot = rel_rot.float()
        bs, num_frames, num_modes, _ = rel_pose.shape
        # rel_pose = rel_pose.transpose(1, 2) # B, M, F, 2

        rot = torch.eye(2).to(rel_pose.device)[None]
        pose = torch.zeros_like(gt_rel_pose)
        pose[:, 0] = rel_pose[:, 0][gt_pose_mode[:, 0] == 1]
        for i in range(1, num_frames):
            rot_cur = rel_rot[:, i-1][gt_pose_mode[:, i-1] == 1]
            rot_mat = torch.cat([torch.cos(rot_cur), -torch.sin(rot_cur), torch.sin(rot_cur), torch.cos(rot_cur)], dim=-1).reshape(bs, 2, 2)
            rot = torch.matmul(rot, rot_mat)
            pose_cur = rel_pose[:, i][gt_pose_mode[:, i] == 1]
            pose[:, i] = torch.matmul(rot, pose_cur[..., None])[..., 0]
        pose = torch.cumsum(pose, 1)
        
        # gt_pose_mode = gt_pose_mode.transpose(1,2) # B, F, M -> B, M, F
        # gt_rel_pose = gt_rel_pose.unsqueeze(1).repeat(1, num_modes, 1, 1) # B, M, F, 2
        gt_rel_pose = torch.cumsum(gt_rel_pose, 1)
            
        if self.loss_type == 'l1':
            loss = torch.abs(pose - gt_rel_pose).sum(-1)
        elif self.loss_type == 'l2':
            loss = torch.sqrt(((pose - gt_rel_pose) ** 2).sum(-1))
        
        return loss.mean(0)