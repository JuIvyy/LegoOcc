import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
import math
from .base_loss import BaseLoss
from . import GPD_LOSS

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - math.log(1 + depth_min))
            / (math.log(1 + depth_max) - math.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds (-2, -1, 0, 1, ..., num_bins, num_bins +1) --> (num_bins, num_bins, 0, 1, ..., num_bins, num_bins)
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices.long()

@GPD_LOSS.register_module()
class DepthLoss(BaseLoss):
    def __init__(self, weight=1.0, downsample_factor=None, d_bound=None, depth_channels=64, input_dict=None, loss_name='depth_loss'):
        super().__init__(weight)

        self.loss_name = loss_name
        self.input_dict = input_dict
        self.downsample_factor = downsample_factor
        self.depth_channels = depth_channels
        self.d_bound = d_bound
        self.loss_func = self.get_depth_loss
        
    def get_depth_loss(self, depth_labels, depth_preds, weight=None):
        # Ensure both are float tensors
        depth_labels = depth_labels.to(dtype=torch.float32)
        depth_preds = depth_preds.to(dtype=torch.float32)

        # Resize prediction to match label resolution
        if depth_preds.shape[-2:] != depth_labels.shape[-2:]:
            depth_preds = F.interpolate(
                depth_preds, size=depth_labels.shape[-2:], mode='bilinear', align_corners=True
            )

        if weight is not None and weight.shape[-2:] != depth_labels.shape[-2:]:
            weight = F.interpolate(
                weight, size=depth_labels.shape[-2:], mode='nearest',
            )

        # Create valid mask: ignore invalid (<= 0) ground-truth
        valid_mask = (depth_labels > 0).to(dtype=torch.float32)  # (B, 1, H, W)
        if weight is not None:
            valid_mask = valid_mask * weight.squeeze(1)

        # Compute Huber loss per-pixel
        per_pixel_loss = F.huber_loss(depth_preds, depth_labels, reduction='none', delta=1.0)  # (B, 1, H, W)

        # Apply mask
        depth_loss = (per_pixel_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        return depth_loss
        

@GPD_LOSS.register_module()
class DepthClsLoss(BaseLoss):
    def __init__(self, weight=1.0, downsample_factor=None, d_bound=None, depth_channels=64, input_dict=None):
        super().__init__(weight)
        self.input_dict = input_dict
        self.downsample_factor = downsample_factor
        self.depth_channels = depth_channels
        self.d_bound = d_bound
        self.loss_func = self.get_depth_loss

    def _get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample_factor * self.downsample_factor)

        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(
            B * N, H // self.downsample_factor, W // self.downsample_factor
        )

        # ds = torch.arange(64)  # (64,)
        # depth_bin_pos = (10.0 / 64 / 65 * ds * (ds + 1)).reshape(1, 64, 1, 1)
        # # print(gt_depths.unsqueeze(1).shape)
        # # print(depth_bin_pos.shape)
        # delta_z = torch.abs(gt_depths.unsqueeze(1) - depth_bin_pos.to(gt_depths.device))
        # gt_depths = torch.argmin(delta_z, dim=1)
        gt_depths = bin_depths(gt_depths, mode='LID', depth_min=0, depth_max=10, num_bins=self.depth_channels, target=True)

        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1
        ).view(-1, self.depth_channels + 1)[:, :-1]

        return gt_depths.float()

    def get_depth_loss(self, depth_labels, depth_preds):
        if len(depth_labels.shape) != 4:
            depth_labels = depth_labels.unsqueeze(1)
            # print(depth_labels.shape)
        N_pred, n_cam_pred, D, H, W = depth_preds.shape
        N_gt, n_cam_label, oriH, oriW = depth_labels.shape
        assert (
            N_pred * n_cam_pred == N_gt * n_cam_label
        ), f"N_pred: {N_pred}, n_cam_pred: {n_cam_pred}, N_gt: {N_gt}, n_cam_label: {n_cam_label}"
        depth_labels = depth_labels.reshape(N_gt * n_cam_label, oriH, oriW)
        depth_preds = depth_preds.reshape(N_pred * n_cam_pred, D, H, W)

        # depth_labels = depth_labels.reshape(
        #     N
        # )

        # depth_labels = depth_labels.unsqueeze(1)
        # depth_labels = depth_labels
        depth_labels = F.interpolate(
            depth_labels.unsqueeze(1),
            (H * self.downsample_factor, W * self.downsample_factor),
            mode="nearest",
        )
        depth_labels = self._get_downsampled_gt_depth(depth_labels)
        depth_preds = (
            depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        )
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction="none",
            ).sum() / max(1.0, fg_mask.sum())
            # depth_loss = torch.nan_to_num(depth_loss, nan=0.)

        return depth_loss