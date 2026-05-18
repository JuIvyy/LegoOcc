# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from .base_loss import BaseLoss
from . import GPD_LOSS
import torch
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.
    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    assert target.min() > 0, 'hard code so ignore first class'
    target = F.one_hot(target, num_classes=num_classes + 1)

    target = target[:, :num_classes]
    # target = target[:, 1:num_classes]
    # pred = pred[:, 1:]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)

    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight

    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None,
                       activated=False):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    # print('pred:', pred)
    # print('target:', target)

    if activated:
        loss = py_focal_loss_with_prob(pred, target, None, gamma, alpha,
                               reduction, avg_factor)
    else:
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')

    if weight is not None:

        # # hard code, jst rm first class
        # weight = weight[:, 1:]

        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)

        assert weight.ndim == loss.ndim
        loss = loss * weight

    if reduction != 'none':
        loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@GPD_LOSS.register_module()
class FocalLoss(BaseLoss):

    def __init__(self, weight=1.0, gamma=2.0, alpha=0.25, ignore_label=255,
                 cls_freq=None, input_dict=None, activated=False, **kwargs):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
        """
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'target': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_label = ignore_label
        self.cls_weight = torch.from_numpy(1 / np.log(cls_freq)).cuda()
        # self.cls_weight = torch.cat([self.cls_weight, torch.tensor([1])]).cuda()
        
        H, W = 256, 256       # hard coding
        xy, yx = torch.meshgrid([torch.arange(H)-H/2,  torch.arange(W)-W/2])
        c = torch.stack([xy,yx], 2)
        c = torch.norm(c, 2, -1)
        c_max = c.max()
        self.c = (c/c_max + 1).cuda()
        self.activated = activated

    def focal_loss(self, pred, target, fov_mask, reduction='mean'):
        pred = pred.float()
        target = target.long()

        B, H, W, D = target.shape
        # c = self.c[None, :, :, None].repeat(B, 1, 1, D).reshape(-1)
        c = torch.ones_like(target).reshape(-1).cuda() # 129600

        visible_mask = ((target!=self.ignore_label) & fov_mask).reshape(-1).nonzero().squeeze(-1)
        weight_mask = self.cls_weight[None,:] * c[visible_mask, None]
        # visible_mask[:, None]

        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]
        target = target.reshape(-1)[visible_mask]
        # if target.numel() == 0:
        #     loss_cls = torch.tensor(0.0, device=pred.device)
        # else:
        #     loss_cls = sigmoid_focal_loss(
        #         pred,
        #         target,
        #         weight_mask,
        #         gamma=self.gamma,
        #         alpha=self.alpha)

        loss_cls = sigmoid_focal_loss(
            pred,
            target,
            weight_mask,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            activated=self.activated)
        return loss_cls
    

@GPD_LOSS.register_module()
class GlobalFocalLoss(BaseLoss):

    def __init__(self, weight=1.0, gamma=2.0, alpha=0.25, ignore_label=255,
                 cls_freq=None, input_dict=None, **kwargs):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
        """
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'target': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_label = ignore_label
        self.cls_weight = torch.from_numpy(1 / np.log(cls_freq)).cuda()
        # self.cls_weight = torch.cat([self.cls_weight, torch.tensor([1])]).cuda()
        
        H, W = 256, 256       # hard coding
        xy, yx = torch.meshgrid([torch.arange(H)-H/2,  torch.arange(W)-W/2])
        c = torch.stack([xy,yx], 2)
        c = torch.norm(c, 2, -1)
        c_max = c.max()
        self.c = (c/c_max + 1).cuda()
  
        
    def focal_loss(self, pred, target):
        pred = pred.float()
        target = target.long()

        B, H, W, D = target.shape
        # c = self.c[None, :, :, None].repeat(B, 1, 1, D).reshape(-1)
        c = torch.ones_like(target).reshape(-1).cuda() # 129600
        
        visible_mask = (target!=self.ignore_label).reshape(-1).nonzero().squeeze(-1)
        weight_mask = self.cls_weight[None,:] * c[visible_mask, None]
        # visible_mask[:, None]

        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]
        target = target.reshape(-1)[visible_mask]
        # if target.numel() == 0:
        #     loss_cls = torch.tensor(0.0, device=pred.device)
        # else:
        #     loss_cls = sigmoid_focal_loss(
        #         pred,
        #         target,
        #         weight_mask,
        #         gamma=self.gamma,
        #         alpha=self.alpha)
        loss_cls = sigmoid_focal_loss(
            pred,
            target,
            weight_mask,
            gamma=self.gamma,
            alpha=self.alpha)
        return loss_cls


@GPD_LOSS.register_module()
class BinaryFocalLossProb(BaseLoss):

    def __init__(self, weight=1.0, gamma=2.0, alpha=0.25, ignore_label=255,
                 cls_freq=None, input_dict=None, activated=True, **kwargs):
        """
        这里 activated=True 表示 pred 已是概率（我们不再做激活），与“probability 输入”一致
        """
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {'pred': 'ce_input', 'target': 'ce_label'}
        else:
            self.input_dict = input_dict

        self.loss_func = self.focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_label = ignore_label
        self.activated = activated  # 对齐你的标志位

        assert cls_freq is not None, "需要传入 cls_freq 以构造类别权重"
        self.register_buffer('cls_weight', torch.from_numpy(1 / np.log(cls_freq + 1e-6)).float())

        # 可选：中心权重（你原来的 c），如不需要可注释
        H, W = 256, 256   # 若你的分辨率可变，建议在 forward 里按真实 H,W 构造
        xy, yx = torch.meshgrid(torch.arange(H) - H/2, torch.arange(W) - W/2, indexing='ij')
        c = torch.stack([xy, yx], dim=2).norm(dim=-1)  # (H, W)
        c = (c / (c.max() + 1e-6) + 1).float()
        self.register_buffer('c_map', c)

    def sigmoid_focal_loss_prob(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Focal Loss (RetinaNet) 的概率输入版本：
        - probs:   已是概率(0~1)，与 targets 同形状
        - targets: 二值标签(0/1)，与 probs 同形状

        与 logits 版本的差异：
        - 取消 sigmoid；用 BCE(概率) 代替 BCE-with-logits
        - 其它公式保持一致：p_t、(1 - p_t)^gamma、alpha 权重

        Args:
            probs (Tensor): 已经是概率分布，形状任意，但与 targets 一致。
            targets (Tensor): 二值标签(0/1)，与 probs 同形状。
            alpha (float):   [0,1] 内的正负样本平衡系数，或 -1 表示忽略 alpha。
            gamma (float):   调制因子指数。
            reduction (str): 'none' | 'mean' | 'sum'
            eps (float):     数值稳定性用的下界裁剪。

        Returns:
            Tensor: 根据 reduction 返回对应标量或张量。
        """
        if not (0 <= alpha <= 1) and alpha != -1:
            raise ValueError(f"Invalid alpha value: {alpha}. Must be in [0,1] or -1.")

        if probs.shape != targets.shape:
            raise ValueError(f"probs and targets must have the same shape, got {probs.shape} vs {targets.shape}")

        # 数值稳定：避免 log(0)
        probs = probs.clamp(min=eps, max=1.0 - eps)

        # BCE(概率版)
        ce_loss = F.binary_cross_entropy(probs, targets, reduction="none")

        # p_t = p  (if y=1)  或 1-p (if y=0)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # 调制项
        loss = ce_loss * ((1.0 - p_t) ** gamma)

        # alpha 平衡
        if alpha >= 0:
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * loss

        # 规约
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid reduction: '{reduction}'. Supported: 'none', 'mean', 'sum'."
            )