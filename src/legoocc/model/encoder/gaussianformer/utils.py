import torch.nn as nn, torch
from typing import NamedTuple
from torch import Tensor
import torch.nn.functional as F

from mmengine import MODELS
from mmengine.model import BaseModule, Sequential
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout

SIGMOID_MAX = 4.595 * 2


class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor
    harmonics: Tensor = None
    feat: Tensor = None
    conf: Tensor = None


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)

def batch_quaternion_multiply(q_cam2world, q_cam_list):
    """
    对一组四元数进行批量复合运算。
    
    参数:
    q_cam2world: torch tensor of shape (4,)
        相机到世界坐标系的四元数，格式为 [w, x, y, z]。
    q_cam_list: torch tensor of shape (N, 4)
        相机坐标系下的 N 个四元数，格式为 [[w, x, y, z], ...]。
        
    返回值:
    torch tensor of shape (N, 4)
        复合运算后的 N 个四元数，格式为 [[w, x, y, z], ...]。
    """
    # 拆分四元数的实部和虚部
    w1, x1, y1, z1 = q_cam2world  # 形状为 (4,)
    w2, x2, y2, z2 = q_cam_list[:, 0], q_cam_list[:, 1], q_cam_list[:, 2], q_cam_list[:, 3]  # 形状为 (N,)
    
    # 计算乘积的实部和虚部，形状为 (N,)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # 将结果堆叠为 (N, 4) 的张量
    return torch.stack((w, x, y, z), dim=1)

def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4
    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


def safe_get_quaternion(R):
    assert R.shape[-2:] == (3, 3), "Input must be a 3x3 matrix"
    
    four_W_squared = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    four_X_squared = R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]
    four_Y_squared = -R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]
    four_Z_squared = -R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]
    
    max_index = 0
    max_value = four_W_squared
    
    if four_X_squared > max_value:
        max_index = 1
        max_value = four_X_squared
    if four_Y_squared > max_value:
        max_index = 2
        max_value = four_Y_squared
    if four_Z_squared > max_value:
        max_index = 3
        max_value = four_Z_squared
    
    q_max = torch.sqrt(max_value + 1) / 2
    mult = 1 / (4 * q_max)
    
    if max_index == 0:
        q0 = q_max
        q1 = (R[..., 2, 1] - R[..., 1, 2]) * mult
        q2 = (R[..., 0, 2] - R[..., 2, 0]) * mult
        q3 = (R[..., 1, 0] - R[..., 0, 1]) * mult
    elif max_index == 1:
        q0 = (R[..., 2, 1] - R[..., 1, 2]) * mult
        q1 = q_max
        q2 = (R[..., 1, 0] + R[..., 0, 1]) * mult
        q3 = (R[..., 0, 2] + R[..., 2, 0]) * mult
    elif max_index == 2:
        q0 = (R[..., 0, 2] - R[..., 2, 0]) * mult
        q1 = (R[..., 0, 1] + R[..., 1, 0]) * mult
        q2 = q_max
        q3 = (R[..., 2, 1] + R[..., 1, 2]) * mult
    else:
        q0 = (R[..., 1, 0] - R[..., 0, 1]) * mult
        q1 = (R[..., 0, 2] + R[..., 2, 0]) * mult
        q2 = (R[..., 2, 1] + R[..., 1, 2]) * mult
        q3 = q_max
        
    quaternion = torch.stack((q0, q1, q2, q3), dim=-1)
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    return quaternion


# def safe_get_quaternion(R):
#     """
#     Converts a rotation matrix (or a batch of rotation matrices) to a quaternion.
#     This implementation handles potential numerical instability by picking the
#     largest component to calculate first.

#     Args:
#         R (torch.Tensor): A tensor representing a rotation matrix or a batch of
#                           rotation matrices. Its last two dimensions must be (3, 3).
#                           E.g., (..., 3, 3) where '...' can be any number of batch dimensions.

#     Returns:
#         torch.Tensor: A quaternion or a batch of quaternions, with shape (..., 4),
#                       in [w, x, y, z] order.
#     """
#     assert R.shape[-2:] == (3, 3), "Input must be a (..., 3, 3) tensor."

#     # These will have the same leading batch dimensions as R
#     four_W_squared = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] + 1.0 # Added +1.0 for the derivation
#     four_X_squared = R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2] + 1.0
#     four_Y_squared = -R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2] + 1.0
#     four_Z_squared = -R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2] + 1.0

#     # Stack them to easily find the max for each batch element
#     # Shape will be (..., 4) where 4 corresponds to [w, x, y, z] order for the squared terms
#     four_squared_terms = torch.stack((four_W_squared,
#                                       four_X_squared,
#                                       four_Y_squared,
#                                       four_Z_squared), dim=-1)

#     # Find the max value and its index for each item in the batch
#     # max_value_per_batch: (...,)
#     # max_index_per_batch: (...,) - contains 0, 1, 2, or 3 for each batch element
#     max_value_per_batch, max_index_per_batch = torch.max(four_squared_terms, dim=-1)

#     # Handle potential negative values due to numerical precision (should be very close to zero)
#     # Clip max_value_per_batch to ensure non-negative sqrt.
#     max_value_per_batch = torch.relu(max_value_per_batch) 

#     # Calculate q_max for each batch element
#     q_max = torch.sqrt(max_value_per_batch) / 2.0
    
#     # Handle the case where q_max might be very close to zero (e.g., identity matrix with numerical issues)
#     # This avoids division by zero or large numbers.
#     # We add a small epsilon to the denominator for numerical stability.
#     mult = 1.0 / (4.0 * q_max + 1e-8) # Add a small epsilon for stability

#     # Initialize quaternion components with zeros, then fill based on max_index
#     # The shape will match the leading batch dimensions of R, followed by 4 for quaternion components
#     quaternion = torch.zeros(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)

#     # Use torch.where or advanced indexing to assign components based on max_index_per_batch
#     # This is the core of the batchification for the conditional logic

#     # For max_index == 0 (w is largest)
#     mask_w = (max_index_per_batch == 0)
#     quaternion[mask_w, 0] = q_max[mask_w]
#     quaternion[mask_w, 1] = (R[mask_w, 2, 1] - R[mask_w, 1, 2]) * mult[mask_w]
#     quaternion[mask_w, 2] = (R[mask_w, 0, 2] - R[mask_w, 2, 0]) * mult[mask_w]
#     quaternion[mask_w, 3] = (R[mask_w, 1, 0] - R[mask_w, 0, 1]) * mult[mask_w]

#     # For max_index == 1 (x is largest)
#     mask_x = (max_index_per_batch == 1)
#     quaternion[mask_x, 0] = (R[mask_x, 2, 1] - R[mask_x, 1, 2]) * mult[mask_x]
#     quaternion[mask_x, 1] = q_max[mask_x]
#     quaternion[mask_x, 2] = (R[mask_x, 1, 0] + R[mask_x, 0, 1]) * mult[mask_x]
#     quaternion[mask_x, 3] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) * mult[mask_x]

#     # For max_index == 2 (y is largest)
#     mask_y = (max_index_per_batch == 2)
#     quaternion[mask_y, 0] = (R[mask_y, 0, 2] - R[mask_y, 2, 0]) * mult[mask_y]
#     quaternion[mask_y, 1] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) * mult[mask_y]
#     quaternion[mask_y, 2] = q_max[mask_y]
#     quaternion[mask_y, 3] = (R[mask_y, 2, 1] + R[mask_y, 1, 2]) * mult[mask_y]

#     # For max_index == 3 (z is largest)
#     mask_z = (max_index_per_batch == 3)
#     quaternion[mask_z, 0] = (R[mask_z, 1, 0] - R[mask_z, 0, 1]) * mult[mask_z]
#     quaternion[mask_z, 1] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) * mult[mask_z]
#     quaternion[mask_z, 2] = (R[mask_z, 2, 1] + R[mask_z, 1, 2]) * mult[mask_z]
#     quaternion[mask_z, 3] = q_max[mask_z]

#     # Normalize the quaternion(s) for numerical stability.
#     # It's crucial to normalize after filling, as the individual components
#     # are calculated based on potentially different formulas.
#     # Add a small epsilon to the norm to prevent division by zero in case of an all-zero quaternion
#     norm = torch.norm(quaternion, dim=-1, keepdim=True)
#     quaternion = quaternion / (norm + 1e-8)
    
#     return quaternion
    

def get_quaternion(R):
    """
    Converts a 3x3 rotation matrix to a normalized quaternion.
    
    Parameters:
    R (torch.Tensor): A tensor of shape (..., 3, 3) representing the rotation matrix.
    
    Returns:
    torch.Tensor: A tensor of shape (..., 4) representing the normalized quaternion (a, b, c, d).
    """
    assert R.shape[-2:] == (3, 3), "Input must be a 3x3 matrix"
    
    q0 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])) / 2
    q1 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2
    q2 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2
    q3 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2
    
    # Determine the signs of q1, q2, q3 based on the elements of R
    q1 = torch.copysign(q1, R[..., 2, 1] - R[..., 1, 2])
    q2 = torch.copysign(q2, R[..., 0, 2] - R[..., 2, 0])
    q3 = torch.copysign(q3, R[..., 1, 0] - R[..., 0, 1])
    
    # Stack into a single quaternion tensor
    quaternion = torch.stack((q0, q1, q2, q3), dim=-1)
    
    # Normalize the quaternion to ensure it is unit-length
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    return quaternion

def cartesian(anchor, pc_range):
    xyz = safe_sigmoid(anchor[..., :3])

    if pc_range.ndim == 1:
        xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    elif pc_range.ndim == 2:
        assert pc_range.shape[0] == anchor.shape[0]
        pc_range = pc_range.view(pc_range.shape[0], *[1 for _ in range(anchor.ndim - 2)], pc_range.shape[-1])

        xxx = xyz[..., 0] * (pc_range[..., 3] - pc_range[..., 0]) + pc_range[..., 0]
        yyy = xyz[..., 1] * (pc_range[..., 4] - pc_range[..., 1]) + pc_range[..., 1]
        zzz = xyz[..., 2] * (pc_range[..., 5] - pc_range[..., 2]) + pc_range[..., 2]

    else:
        raise ValueError()

    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz


@MODELS.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
