import torch
import torch.nn as nn

from . import GPD_LOSS


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



@GPD_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(
        self,
        loss_cfgs,
        valid_range=-1,
        ignore_label=0,
        alpha = 0.2,
        gamma = 1.0,
    ):
        super().__init__()
        self.num_losses = len(loss_cfgs)
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(GPD_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.valid_range = valid_range
        self.ignore_label = ignore_label

        self.alpha = alpha
        self.gamma = gamma
        self.weight = sum([loss_cfg.get('weight', 1.0) for loss_cfg in loss_cfgs])

    def forward(self, inputs):
        loss_dict = {}
        tot_loss = 0.

        gamma = self.gamma
        alpha = self.alpha

        force_reduction_none = self.valid_range > 0

        # if 'confs' in inputs:
        #     confs = inputs['confs']
        #     confs = torch.stack(confs)

        #     fov_mask = inputs['fov_mask']
        #     target = inputs['ce_label']
        #     mask = (target != self.ignore_label) & fov_mask

        #     # expp1
        #     confs = 1 + confs.exp()
        #     confs = confs[mask.flatten(1)]
        #     # inputs['weight'] = confs

        #     loss_conf = -alpha * torch.log(confs)[:, None]
        #     # tot_loss += loss_conf
        #     # loss_dict['loss_conf'] = loss_conf.detach().item()
        #     force_reduction_none = True

        # else:
        #     confs = None

        for loss_func in self.losses:

            loss_name = getattr(loss_func, 'loss_name', loss_func.__class__.__name__)

            if 'focal' in loss_name.lower():
                loss = loss_func(
                    inputs,
                    force_reduction_none=force_reduction_none)

                if force_reduction_none:

                    # if confs is not None:
                    #     loss_conf = loss_conf + gamma * loss * confs.view(-1, *(1 for _ in loss.shape[1:]))

                    loss = loss.flatten(1).mean(1)

                    if self.valid_range > 0:
                        loss = filter_by_quantile(loss, self.valid_range)

                    loss = loss.mean()
            else:

                loss = loss_func(inputs)

            tot_loss += loss
            loss_dict.update({
                loss_name: \
                loss.detach().item(), # / loss_func.weight
            })

        # if confs is not None:

        #     loss_conf = loss_conf.flatten(1).mean(1)

        #     if self.valid_range > 0:
        #         loss_conf = filter_by_quantile(loss_conf, self.valid_range)

        #     loss_conf = loss_conf.mean()
        #     tot_loss += loss_conf
        #     loss_dict['loss_conf'] = loss_conf.cpu().item()

        return tot_loss, loss_dict
