import torch
import torch.nn.functional as F
import torch_scatter
from gsplat import rasterization
import numpy as np
from einops import einsum, rearrange, repeat

from .utils import unbatched_forward


def grad_nan_to_num_hook(grad):
    return torch.nan_to_num(grad)



@unbatched_forward
def rasterize_gaussians(means3d,
                        colors,
                        opacities,
                        scales,
                        rotations,
                        cam2imgs,
                        cam2egos,
                        image_size,
                        img_aug_mats=None,
                        radius_clip=0.5,
                        **kwargs):

    # cam2world to world2cam
    R = cam2egos[:, :3, :3].mT
    T = -R @ cam2egos[:, :3, 3:4]
    viewmat = torch.zeros_like(cam2egos)
    viewmat[:, :3, :3] = R
    viewmat[:, :3, 3:] = T
    viewmat[:, 3, 3] = 1

    if isinstance(cam2imgs, np.ndarray):
        cam2imgs = torch.from_numpy(cam2imgs).to(cam2egos)

    if cam2imgs.shape[-2:] == (4, 4):
        cam2imgs = cam2imgs[:, :3, :3]

    if img_aug_mats is not None:
        cam2imgs = cam2imgs.clone()
        cam2imgs[:, :2, :2] *= img_aug_mats[:, :2, :2]
        image_size = image_size.tolist()

        for i in range(2):
            cam2imgs[:, i, 2] *= img_aug_mats[:, i, i]
            cam2imgs[:, i, 2] += img_aug_mats[:, i, 3]
            image_size[1 - i] = round(image_size[1 - i] *
                                      img_aug_mats[0, i, i].item() +
                                      img_aug_mats[0, i, 3].item())

    rendered_image, alpha, _ = rasterization(
        means=means3d,
        quats=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=cam2imgs,
        width=image_size[1],
        height=image_size[0],
        radius_clip=radius_clip,
        **kwargs)

    return rendered_image.permute(0, 3, 1, 2), alpha.squeeze(-1)

