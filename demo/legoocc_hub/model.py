"""Self-contained LegoOcc inference model (no mm* dependencies).

Usage:
    from legoocc_hub import LegoOcc
    model = LegoOcc.from_pretrained("JuIvyy/LegoOcc").cuda().eval()
    out = model.predict("room.jpg", intrinsics=K)
    model.set_vocabulary(["plant", "lamp", ...])   # open vocabulary (needs open_clip_torch)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin

from . import ops
from .depth_anything_v2.dpt import DepthAnythingV2
from .msda import MultiScaleDeformableAttention
from .utils import (DEFAULT_VOCABULARY, default_cam2world, default_intrinsics, default_vox_origin,
                    get_rotation_matrix, resize_for_model, safe_sigmoid)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class LegoOcc(
    nn.Module,
    PyTorchModelHubMixin,
    library_name='legoocc',
    tags=['occupancy-prediction', 'open-vocabulary', '3d-gaussians', 'monocular', 'cvpr2026'],
    license='cc-by-nc-4.0',
    pipeline_tag='image-to-3d',
):
    """Monocular open-vocabulary occupancy with language-embedded 3D Gaussians.

    Args:
        vocabulary: class prompts whose CLIP text features are stored with the weights.
        num_bins: number of Gaussians sampled along each anchor ray.
        min_temperature: opacity temperature used at inference.
        grid_shape / voxel_size: output occupancy grid (default 60x60x36 @ 8cm).
    """

    def __init__(
        self,
        vocabulary=tuple(DEFAULT_VOCABULARY),
        num_bins=8,
        scale_range=(0.01, 0.08),
        opacities_threshold=0.01,
        min_temperature=0.001,
        max_depth=8.0,
        ov_feat_dim=512,
        patch_size=14,
        grid_shape=(60, 60, 36),
        voxel_size=0.08,
        scale_multiplier=3,
    ):
        super().__init__()
        self.vocabulary = list(vocabulary)
        self.num_bins = num_bins
        self.scale_range = tuple(scale_range)
        self.opacities_threshold = opacities_threshold
        self.min_temperature = min_temperature
        self.max_depth = max_depth
        self.patch_size = patch_size
        self.grid_shape = tuple(grid_shape)
        self.voxel_size = voxel_size
        self.scene_size = tuple(s * voxel_size for s in grid_shape)

        self.register_buffer('_resnet_mean', torch.tensor(_RESNET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer('_resnet_std', torch.tensor(_RESNET_STD).view(1, 1, 3, 1, 1), persistent=False)
        # CLIP text features of `vocabulary`, (K, ov_feat_dim)
        self.register_buffer('text_features', torch.zeros(len(self.vocabulary), ov_feat_dim))

        self.backbone = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], max_depth=20)
        self.gs_head = self.backbone.depth_head
        self.backbone.depth_head = None
        self.gs_head.scratch.refinenet4.resConfUnit1 = None
        self.gs_head.scratch.output_conv2 = None
        bkb_feat_dim = self.backbone.pretrained.embed_dim

        self.aggregator = ops.LocalAggregator(
            scale_multiplier=scale_multiplier,
            H=grid_shape[0], W=grid_shape[1], D=grid_shape[2],
            pc_min=[-51.2, -51.2, -5.0],
            grid_size=voxel_size)

        _dim_ = 256
        self.gs_pred_layer = nn.Sequential(
            nn.Linear(_dim_, _dim_), nn.ReLU(),
            nn.Linear(_dim_, _dim_), nn.ReLU(),
            nn.Linear(_dim_, _dim_), nn.ReLU(),
            nn.Linear(_dim_, 3 + 4 + 1),
        )
        self.anchor_feat_layer = nn.Sequential(
            nn.Conv2d(64, _dim_, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(_dim_, _dim_, kernel_size=2, stride=2),
        )
        self.anchor_depth_pred = nn.Sequential(
            nn.Conv2d(_dim_, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )
        self.bins_emb = nn.Embedding(num_bins, _dim_)
        self.depth_scale_layer = nn.Conv2d(_dim_, 1, kernel_size=1)

        self.feat2query = nn.Sequential(
            nn.Linear(_dim_, _dim_), nn.ReLU(), nn.Linear(_dim_, ov_feat_dim))
        self.xyz2qpos = nn.Linear(3, ov_feat_dim)
        self.kv_proj = nn.Sequential(
            nn.Linear(bkb_feat_dim, ov_feat_dim), nn.ReLU(), nn.Linear(ov_feat_dim, ov_feat_dim))
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=ov_feat_dim, num_heads=4, num_points=4, num_levels=4, dropout=0.1)
        self.ov_feat_post_layer = nn.Sequential(
            nn.Linear(ov_feat_dim, ov_feat_dim), nn.ReLU(), nn.Linear(ov_feat_dim, ov_feat_dim))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        # Build (or load the cached) CUDA op right away so the first predict() is fast.
        if torch.cuda.is_available():
            ops.try_load_ext()
        return model

    def set_vocabulary(self, vocabulary):
        """Replace the class prompts, e.g. ["chair", "potted plant", "lamp"] (needs open_clip_torch)."""
        from .text import encode_text
        self.text_features = encode_text(list(vocabulary), device=self.text_features.device)
        self.vocabulary = list(vocabulary)

    # ------------------------------------------------------------------ #
    # Gaussian prediction (camera frame)
    # ------------------------------------------------------------------ #
    def predict_gaussians(self, images, intrinsics):
        """
        Args:
            images: (B, 3, H, W) RGB in [0, 1]; H, W multiples of 14 (long side 518 recommended).
            intrinsics: (B, 3, 3) for the given image resolution.
        Returns:
            dict of Gaussians in the camera frame (x right, y down, z forward):
            means (B, G, 3), scales (B, G, 3), rotations (B, G, 4, wxyz),
            opacities (B, G, 1), features (B, G, 512) CLIP-aligned language features.
        """
        # The released model was trained on BGR inputs normalized with RGB ImageNet stats.
        imgs = images.flip(1)[:, None]
        imgs = (imgs - self._resnet_mean) / self._resnet_std

        features, patch_h, patch_w = self.backbone.custom_forward(imgs.flatten(0, 1))
        mlvl_img_feats = [t[0] for t in features]
        anchor_feat = self.gs_head.custom_forward(features, patch_h, patch_w)
        anchor_feat = self.anchor_feat_layer(anchor_feat)
        anchor_depth = self.anchor_depth_pred(anchor_feat).sigmoid() * self.max_depth

        device, dtype = anchor_depth.device, anchor_depth.dtype
        B, _, H, W = anchor_depth.shape
        iH, iW = imgs.shape[-2:]

        u, v = torch.meshgrid(
            torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
        intr = intrinsics.to(dtype=dtype, device=device)
        fx = intr[:, 0, 0].view(B, 1, 1)
        fy = intr[:, 1, 1].view(B, 1, 1)
        cx = intr[:, 0, 2].view(B, 1, 1)
        cy = intr[:, 1, 2].view(B, 1, 1)
        x = (u[None].to(dtype) / W * iW - cx) / fx
        y = (v[None].to(dtype) / H * iH - cy) / fy

        depth_scale = self.depth_scale_layer(anchor_feat).sigmoid()
        depth_offset = torch.linspace(
            0, 5, self.num_bins, device=device, dtype=dtype)[None, :, None, None] * depth_scale
        Z = anchor_depth + depth_offset
        X = repeat(x, 'b h w -> b k h w', k=self.num_bins) * Z
        Y = repeat(y, 'b h w -> b k h w', k=self.num_bins) * Z
        xyz = rearrange(torch.stack([X, Y, Z], dim=2), 'b n c h w -> b (n h w) c')
        feat = anchor_feat.unsqueeze(1) + self.bins_emb.weight[None, :, :, None, None]
        feat = rearrange(feat, 'b n c h w -> b (n h w) c')

        gs_feat = self.gs_pred_layer(feat)

        # language features: deformable attention from each Gaussian into the backbone tokens
        ref_pt = torch.stack([u / W, v / H], dim=-1).float()
        ref_pt = repeat(ref_pt, 'h w c -> b (n h w) l c', b=B, n=self.num_bins, l=len(mlvl_img_feats))
        fh, fw = iH // self.patch_size, iW // self.patch_size
        spatial_shapes = torch.as_tensor([[fh, fw]] * len(mlvl_img_feats), dtype=torch.long, device=device)
        kv = self.kv_proj(torch.cat(mlvl_img_feats, dim=1))
        ov_feat = self.deform_attn(
            query=self.feat2query(feat), value=kv, query_pos=self.xyz2qpos(xyz),
            reference_points=ref_pt, spatial_shapes=spatial_shapes)
        ov_feat = self.ov_feat_post_layer(ov_feat)

        scales = safe_sigmoid(gs_feat[..., :3])
        scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scales
        return dict(
            means=xyz,
            scales=scales,
            rotations=gs_feat[..., 3:7],
            opacities=torch.sigmoid(gs_feat[..., 7:8] / self.min_temperature),
            features=ov_feat,
        )

    # ------------------------------------------------------------------ #
    # Voxelization (world frame)
    # ------------------------------------------------------------------ #
    def voxel_centers(self, vox_origin):
        """vox_origin: (B, 3) -> (B, X, Y, Z, 3) voxel centers in world frame."""
        idx = torch.stack(torch.meshgrid(
            *[torch.arange(n, device=vox_origin.device) for n in self.grid_shape],
            indexing='ij'), dim=-1).float()
        return vox_origin[:, None, None, None].float() + (idx + 0.5) * self.voxel_size

    @torch.no_grad()
    def voxelize(self, gaussians, cam2world, vox_origin):
        """
        Args:
            gaussians: output of `predict_gaussians`.
            cam2world: (B, 4, 4) camera-to-world transform of a gravity-aligned (z-up) frame.
            vox_origin: (B, 3) world coordinate of the grid corner (voxel (0, 0, 0)).
        Returns:
            empty_prob (B, X, Y, Z): probability that a voxel is free space.
            features (B, 512, X, Y, Z): splatted language features (not normalized).
        """
        means = gaussians['means']
        device = means.device
        B = means.shape[0]
        cam2world = cam2world.to(device).float()
        vox_origin = vox_origin.to(device).float()
        pc_max = vox_origin + torch.tensor(self.scene_size, device=device)
        centers = self.voxel_centers(vox_origin)

        means_world = torch.einsum('bnk,bjk->bjn', cam2world, F.pad(means, (0, 1), value=1))[..., :3]
        scales = gaussians['scales']
        M = torch.diag_embed(scales) @ get_rotation_matrix(gaussians['rotations'])
        cov = M.transpose(-1, -2) @ M
        rot = cam2world[:, None, :3, :3]
        cov = rot @ cov @ rot.transpose(-1, -2)
        cov_inv = cov.double().inverse().to(cov.dtype)
        opas = gaussians['opacities']
        # language features plus a constant channel, splatted together
        payload = torch.cat([gaussians['features'], torch.ones_like(opas)], dim=-1)
        C = gaussians['features'].shape[-1]

        empty_prob, features = [], []
        eps = 1e-3
        for i in range(B):
            mask = ((means_world[i] > vox_origin[i] + eps) & (means_world[i] < pc_max[i] - eps)).all(-1)
            mask = mask & (opas[i].squeeze(-1) > self.opacities_threshold)
            pts = centers[i].flatten(0, 2)
            if mask.sum() == 0:
                feat = torch.zeros(pts.shape[0], C, device=device)
                bin_logit = torch.zeros_like(feat[:, 0])
            else:
                with torch.cuda.device(device):
                    semantic, bin_logit, _ = self.aggregator(
                        pts, means_world[i][mask], opas[i][mask], payload[i][mask],
                        scales[i][mask], cov_inv[i][mask], vox_origin[i])
                feat = semantic[:, :C]
            empty_prob.append((1 - bin_logit).unflatten(0, self.grid_shape))
            features.append(feat.T.unflatten(-1, self.grid_shape))
        return torch.stack(empty_prob), torch.stack(features)

    def classify(self, empty_prob, features, text_features=None):
        """Label voxels: 0 = empty, k = vocabulary[k - 1]. Also returns cosine similarities."""
        text_features = self.text_features if text_features is None else text_features
        feats = features / features.norm(dim=1, keepdim=True).clamp(min=1e-12)
        similarity = torch.einsum('bchwd,kc->bkhwd', feats, text_features.to(feats))
        labels = similarity.argmax(1) + 1
        labels[empty_prob > 0.5] = 0
        return labels, similarity

    def forward(self, images, intrinsics, cam2world, vox_origin):
        gaussians = self.predict_gaussians(images, intrinsics)
        empty_prob, features = self.voxelize(gaussians, cam2world, vox_origin)
        labels, similarity = self.classify(empty_prob, features)
        return labels, dict(empty_prob=empty_prob, similarity=similarity, features=features, gaussians=gaussians)

    # ------------------------------------------------------------------ #
    # One-call API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, image, intrinsics=None, cam2world=None, vox_origin=None, vocabulary=None,
                pitch_deg=25.0, cam_height=1.5, return_features=False):
        """Predict open-vocabulary occupancy from a single image.

        Args:
            image: file path, PIL.Image, or HxWx3 uint8 RGB numpy array.
            intrinsics: 3x3 camera matrix at the original image resolution.
                Defaults to a ScanNet-like FoV (fx = 0.9 * width) if omitted.
            cam2world: 4x4 camera-to-world pose in a gravity-aligned, z-up frame.
                Defaults to a camera `cam_height` m above the floor, tilted down
                by `pitch_deg` (the median tilt in ScanNet; use 0 for level shots).
            vox_origin: (3,) world position of the occupancy grid corner. Defaults
                to the OccScanNet convention: box on the floor, in front of the camera.
            vocabulary: optional list of class prompts for this call (needs open_clip_torch);
                defaults to the model's current vocabulary.
            return_features: also return the (512, X, Y, Z) voxel language features.
        Returns:
            dict with
              occupancy (X, Y, Z) long: 0 = empty, k = class_names[k]
              class_names: ['empty'] + vocabulary
              empty_prob (X, Y, Z), similarity (K, X, Y, Z), fov_mask (X, Y, Z) bool,
              voxel_centers (X, Y, Z, 3), gaussians (camera frame), cam2world, vox_origin,
              intrinsics (resized), image (BGR float)
        """
        image_rgb = _load_rgb(image)
        H, W = image_rgb.shape[:2]
        if intrinsics is None:
            intrinsics = default_intrinsics(H, W)
        if cam2world is None:
            cam2world = default_cam2world(pitch_deg, cam_height)
        if vox_origin is None:
            vox_origin = default_vox_origin(cam2world, self.scene_size, cam_height)

        image_bgr, K = resize_for_model(image_rgb[..., ::-1], intrinsics)
        device = self.text_features.device
        img = torch.from_numpy(np.ascontiguousarray(image_bgr[..., ::-1])).permute(2, 0, 1)[None] / 255.
        K_t = torch.from_numpy(K).float()[None].to(device)
        T = torch.as_tensor(np.asarray(cam2world), dtype=torch.float32)[None].to(device)
        origin = torch.as_tensor(np.asarray(vox_origin), dtype=torch.float32)[None].to(device)

        gaussians = self.predict_gaussians(img.to(device), K_t)
        empty_prob, features = self.voxelize(gaussians, T, origin)
        text_features = None
        if vocabulary is not None:
            from .text import encode_text
            text_features = encode_text(list(vocabulary), device=device)
        labels, similarity = self.classify(empty_prob, features, text_features)
        centers = self.voxel_centers(origin)[0]
        out = dict(
            occupancy=labels[0],
            class_names=['empty'] + list(vocabulary if vocabulary is not None else self.vocabulary),
            empty_prob=empty_prob[0],
            similarity=similarity[0],
            fov_mask=_fov_mask(centers, T[0], K_t[0], image_bgr.shape[:2]),
            voxel_centers=centers,
            gaussians={k: v[0] for k, v in gaussians.items()},
            cam2world=T[0],
            vox_origin=origin[0],
            intrinsics=K_t[0],
            image=image_bgr,
        )
        if return_features:
            out['features'] = features[0]
        return out


def _load_rgb(image):
    from PIL import Image, ImageOps
    if isinstance(image, str):
        image = ImageOps.exif_transpose(Image.open(image))
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert('RGB'))
    image = np.asarray(image)
    assert image.ndim == 3 and image.shape[-1] == 3, 'expected an HxWx3 RGB image'
    return image


def _fov_mask(centers, cam2world, K, hw):
    pts = centers.reshape(-1, 3)
    world2cam = torch.linalg.inv(cam2world)
    cam = pts @ world2cam[:3, :3].T + world2cam[:3, 3]
    z = cam[:, 2]
    uv = (cam @ K.T)[:, :2] / z.clamp(min=1e-6)[:, None]
    H, W = hw
    mask = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return mask.reshape(centers.shape[:3])
