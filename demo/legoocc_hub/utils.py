import math

import numpy as np
import torch
import torch.nn.functional as F

SIGMOID_MAX = 4.595 * 2

# Default open-vocabulary prompts (the OccScanNet classes); 'television' is the prompt for 'tvs'.
DEFAULT_VOCABULARY = [
    'ceiling', 'floor', 'wall', 'window', 'chair', 'bed',
    'sofa', 'table', 'television', 'furniture', 'objects',
]


def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)


def get_rotation_matrix(tensor):
    """(..., 4) quaternion (w, x, y, z) -> (..., 3, 3) rotation matrix."""
    assert tensor.shape[-1] == 4
    tensor = F.normalize(tensor, dim=-1)
    w, x, y, z = tensor.unbind(-1)
    mat1 = torch.stack([
        torch.stack([w, -x, -y, -z], -1),
        torch.stack([x, w, -z, y], -1),
        torch.stack([y, z, w, -x], -1),
        torch.stack([z, -y, x, w], -1),
    ], -2)
    mat2 = torch.stack([
        torch.stack([w, -x, -y, -z], -1),
        torch.stack([x, w, z, -y], -1),
        torch.stack([y, -z, w, x], -1),
        torch.stack([z, y, -x, w], -1),
    ], -2)
    result = mat1 @ mat2.transpose(-1, -2)
    return result[..., 1:, 1:]


def resize_for_model(image_bgr, intrinsics, target_size=518, patch_size=14):
    """Resize like the OccScanNet training pipeline (long side -> 518, short side
    rounded to a multiple of 14) and rescale the intrinsics accordingly."""
    import cv2

    H, W = image_bgr.shape[:2]
    if W >= H:
        new_W = target_size
        new_H = round(H * (new_W / W) / patch_size) * patch_size
    else:
        new_H = target_size
        new_W = round(W * (new_H / H) / patch_size) * patch_size
    image = cv2.resize(image_bgr.astype(np.float32), (new_W, new_H), interpolation=cv2.INTER_CUBIC)

    K = np.array(intrinsics, dtype=np.float64).copy()[:3, :3]
    K[0, 0] *= new_W / W
    K[0, 2] *= new_W / W
    K[1, 1] *= new_H / H
    K[1, 2] *= new_H / H
    return image, K


def default_intrinsics(H, W):
    # ScanNet-like field of view (fx ~= 0.9 * width, ~58 deg horizontal).
    f = 0.9 * W
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)


def default_cam2world(pitch_deg=25.0, cam_height=1.5):
    """Gravity-aligned world frame (z up) for a camera tilted down by `pitch_deg`.

    The camera sits at (0, 0, cam_height) looking along +y; the floor is z=0.
    """
    t = math.radians(pitch_deg)
    right = [1.0, 0.0, 0.0]
    down = [0.0, -math.sin(t), -math.cos(t)]
    forward = [0.0, math.cos(t), -math.sin(t)]
    T = np.eye(4)
    T[:3, :3] = np.array([right, down, forward]).T
    T[:3, 3] = [0.0, 0.0, cam_height]
    return T


def default_vox_origin(cam2world, scene_size, cam_height=1.5):
    """Place the occupancy box like OccScanNet does: bottom on the floor, the
    camera at the middle of the box edge facing it, box extending forward."""
    cam2world = np.asarray(cam2world, dtype=np.float64)
    cam_pos = cam2world[:3, 3]
    forward = cam2world[:3, 2].copy()
    forward[2] = 0
    axis = int(np.argmax(np.abs(forward[:2])))
    step = np.zeros(3)
    step[axis] = np.sign(forward[axis]) * scene_size[axis] / 2
    center_xy = cam_pos[:2] + step[:2]
    origin = np.array([
        center_xy[0] - scene_size[0] / 2,
        center_xy[1] - scene_size[1] / 2,
        cam_pos[2] - cam_height,
    ])
    return np.round(origin.astype(np.float32), 4)
