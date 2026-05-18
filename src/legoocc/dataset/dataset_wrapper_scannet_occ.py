import numpy as np
import torch
from torch.utils import data
from . import OPENOCC_DATAWRAPPER
from legoocc.dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, ImageAug3D

from .vggt_load_fn import load_and_preprocess_images


img_norm_cfg = dict(
    mean=[t * 255 for t in [0.485, 0.456, 0.406]],
    std=[t * 255 for t in [0.229, 0.224, 0.225]],
    to_rgb=True)


@OPENOCC_DATAWRAPPER.register_module()
class Scannet_Scene_Occ_DatasetWrapper(data.Dataset):
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], phase='train', size_divisor=32):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                ImageAug3D(
                    final_dim=final_dim,
                    resize_lim=resize_lim,
                    is_train=True
                ),
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=size_divisor)
            ]
        else:
            transforms = [
                ImageAug3D(
                    final_dim=final_dim,
                    resize_lim=resize_lim,
                    is_train=False
                ),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=size_divisor)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        imgs, metas, occ = data

        # deal with img augmentation
        F, N, H, W, C = imgs.shape
        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)

        imgs = imgs_dict['img']
        imgs = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)

        FN, C, H, W = imgs.shape
        imgs = imgs.reshape(F, N, C, H, W)
        metas['img_shape'] = imgs_dict['img_shape']
        if imgs_dict.get('img_aug_matrix'):
            img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0)
            metas['img_aug_matrix'] = img_aug_matrix.reshape(F, N, 4, 4)

        for k in ['cam2world', 'vox_origin', 'occ_xyz', 'cam_vox_range', 'world2cam', 'scene_size', 'cam_k', 'fov_mask', 'depth_gt']:
            value = metas[k]

            if isinstance(value, (tuple, list)):
                value = np.array(value)

            value = torch.from_numpy(value.astype(np.float32)) # .cuda()
            metas[k] = value

        data_tuple = (imgs, metas, occ)
        return data_tuple



@OPENOCC_DATAWRAPPER.register_module()
class Scannet_Scene_Occ_DatasetWrapper_VGGT(data.Dataset):
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], phase='train'):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                # ImageAug3D(
                #     final_dim=final_dim,
                #     resize_lim=resize_lim,
                #     is_train=True
                # ),
                PhotoMetricDistortionMultiViewImage(
                    brightness_delta=8,
                    contrast_range=(0.8, 1.2),
                    saturation_range=(0.8, 1.2),
                    hue_delta=4,
                ),
                # NormalizeMultiviewImage(**img_norm_cfg),
                # PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                # ImageAug3D(
                #     final_dim=final_dim,
                #     resize_lim=resize_lim,
                #     is_train=False
                # ),
                # NormalizeMultiviewImage(**img_norm_cfg),
                # PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        imgs, metas, occ = data

        # imgs = load_and_preprocess_images(imgs)
        F, N, H, W, C = imgs.shape

        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}

        for t in self.transforms:
            imgs_dict = t(imgs_dict)

        imgs = np.stack(imgs_dict['img']).reshape(F, N, H, W, C)
        imgs = torch.from_numpy(imgs) / 255.

        cam_intrin = metas['cam_k']

        if 'neighbor_rgbs' in metas:
            metas['neighbor_rgbs'] = torch.from_numpy(np.stack(metas['neighbor_rgbs'])).float()

        # FN, C, H, W = imgs.shape
        # imgs = imgs.reshape(F, N, C, H, W)
        # metas['img_shape'] = imgs_dict['img_shape']

        # if imgs_dict.get('img_aug_matrix'):
        #     img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0)
        #     metas['img_aug_matrix'] = img_aug_matrix.reshape(F, N, 4, 4)
        for k in ['cam2world', 'vox_origin', 'occ_xyz', 'cam_vox_range', 'world2cam', 'scene_size', 'cam_k', 'fov_mask', 'depth_gt']:
            value = metas[k]

            if isinstance(value, (tuple, list)):
                value = np.array(value)

            value = torch.from_numpy(value.astype(np.float32)) # .cuda()
            metas[k] = value

        data_tuple = (imgs, metas, occ)
        return data_tuple
