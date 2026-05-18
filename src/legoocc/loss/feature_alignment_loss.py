import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from . import GPD_LOSS


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


@GPD_LOSS.register_module()
class FeatureAlignmentLoss(nn.Module):
    def __init__(
        self,
        weight=1.0,
        use_density_to_reweight=False,
        first_frame_weight=1.,
        use_conf=False,
        valid_range=-1,
        gamma=1.0,
        alpha=0.2,
    ):
        super().__init__()

        self.weight = weight
        self.loss_name = 'fmalign_loss'
        self.use_density_to_reweight = use_density_to_reweight
        self.first_frame_weight = first_frame_weight
        self.use_conf = use_conf
        self.valid_range = valid_range
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs):

        tgt_feats = inputs['fm_feats'].flatten(0, 2)
        rendered = inputs['rendered'].flatten(0, 1)

        alpha = inputs['rendered_alpha'].clone()
        alpha[:, 0] = alpha[:, 0] * self.first_frame_weight
        alpha = alpha.flatten(0, 1)

        if rendered.shape[-2:] != tgt_feats.shape[-2:]:
            tgt_feats = F.interpolate(tgt_feats, rendered.shape[-2:], mode='bilinear')

        if self.use_density_to_reweight:
            rendered_density = inputs['rendered_density'][:, None].detach()
            if rendered_density.shape[-2:] != tgt_feats.shape[-2:]:
                rendered_density = F.interpolate(rendered_density, tgt_feats.shape[-2:], mode='bilinear')

        tgt_feats = rearrange(tgt_feats, 'b c h w -> (b h w) c')
        rendered = rearrange(rendered, 'b c h w -> (b h w) c')
        alpha = rearrange(alpha, 'b h w -> (b h w)')

        if self.use_density_to_reweight:
            assert False
            rendered_density = rearrange(rendered_density, 'b c h w -> (b h w) c').squeeze(-1)
            loss_cosine = F.cosine_embedding_loss(
                rendered, tgt_feats, torch.ones_like(
                    tgt_feats[:, 0]),
                reduction='none') * self.weight
            loss_cosine = (loss_cosine * loss_cosine).sum() / loss_cosine.sum().clamp(1)
        else:
            loss_cosine = F.cosine_embedding_loss(
                rendered, tgt_feats, torch.ones_like(
                    tgt_feats[:, 0]), reduction='none') * self.weight

            if self.use_conf:
                conf = rearrange(inputs['rendered_conf'].squeeze(2), 'b n h w -> (b n h w)')
                loss_conf = loss_cosine * self.gamma * conf - self.alpha * torch.log(conf)

            if self.valid_range > 0:
                loss_cosine = loss_cosine * alpha

                loss_cosine = filter_by_quantile(loss_cosine, self.valid_range)
                loss_cosine = torch.nan_to_num(loss_cosine).mean()

                if self.use_conf:
                    loss_conf = loss_conf * alpha

                    loss_conf = filter_by_quantile(loss_conf, self.valid_range)
                    loss_conf = torch.nan_to_num(loss_conf).mean()
                    
                    # print(loss_cosine, loss_conf)

                    loss_cosine = loss_cosine + loss_conf
            else:
                # else:
                loss_cosine = (loss_cosine * alpha).sum() / alpha.sum().clamp(1)
                if self.use_conf:
                    loss_conf = (loss_conf * alpha).sum() / alpha.sum().clamp(1)
                    loss_cosine = loss_cosine + loss_conf

        # if loss_cosine < 1e-5:
        #     import pdb; pdb.set_trace()

        if torch.isnan(loss_cosine):
            import pdb; pdb.set_trace()

        return loss_cosine


@GPD_LOSS.register_module()
class RenderMSELoss(nn.Module):
    def __init__(
        self,
        first_frame_weight=1.,
        weight=1.0,
    ):
        super().__init__()

        self.weight = weight
        self.loss_name = 'render_mse_loss'
        self.first_frame_weight = first_frame_weight

    def forward(self, inputs):

        rgb_gt = inputs['rgb_gt']
        rendered_rgb = inputs['rendered_rgb']
        alpha = inputs['rendered_alpha'][:, :, None].clone()

        alpha[:, 0] = alpha[:, 0] * self.first_frame_weight
        
        loss = F.mse_loss(rendered_rgb, rgb_gt, reduction='none') * self.weight
        loss = (loss * alpha).sum() / alpha.sum().clamp(1) / loss.shape[2]

        return loss
