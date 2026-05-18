import os
import json
import glob
import numpy as np
import numba as nb
import torch
from torch.utils import data
import pickle
from PIL import Image
from mmcv.image.io import imread
import copy
from pyquaternion import Quaternion
from . import OPENOCC_DATASET
from legoocc.dataset.nyu_utils import vox2pix
from torchvision import transforms
import math, cv2
from torchvision.transforms import Compose
from tqdm import tqdm
from legoocc.dataset.transform_ import Resize, NormalizeImage, PrepareForNet




@OPENOCC_DATASET.register_module()
class Scannet_Scene_OpenOccupancy_Dataset(data.Dataset):
    def __init__(
        self,
        data_path, 
        num_frames=1,
        new_H=480,
        new_W=640,
        target_size=518,
        offset=0,
        grid_size_occ=[60, 60, 36],
        coarse_ratio=2,
        empty_idx=0,
        phase='train',
        num_pts=21600,
        data_tg='base',
        vggt_image_preprocess=False,
        load_neighbor=0,
        load_neighbor_random_all=0,
        text_path=None,
    ):

        self.occscannet_root = data_path
        self.target_size = target_size
        self.text_path = text_path
        
        self.num_frames = num_frames
        self.offset = offset
        self.grid_size_occ = grid_size_occ
        self.grid_size_occ_coarse = (np.array(grid_size_occ) // coarse_ratio).astype(np.uint32)
        self.coarse_ratio = coarse_ratio
        self.empty_idx = empty_idx
        self.phase = phase

        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        if data_tg == 'base':
            subscenes_list = f'{self.occscannet_root}/{self.phase}_final.txt'
        elif data_tg == 'mini':
            subscenes_list = f'{self.occscannet_root}/{self.phase}_mini_final.txt'
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                self.used_subscenes[i] = f'{self.occscannet_root}/' + self.used_subscenes[i].strip()
        
        self.num_pts = num_pts
        
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.vggt_image_preprocess = vggt_image_preprocess
        self.new_H = new_H
        self.new_W = new_W

        if load_neighbor_random_all:
            assert load_neighbor > 0 and load_neighbor_random_all > load_neighbor
        else:
            load_neighbor_random_all = load_neighbor
        self.load_neighbor_random_all = load_neighbor_random_all

        self.load_neighbor = load_neighbor
        if load_neighbor:
            # pre-build all pose
            self.all_poses = {}
            cache_path = os.path.join(self.occscannet_root, f"{data_tg}_{self.phase}_all_poses_cache.pkl")

            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.all_poses = pickle.load(f)
            else:
                for subscene_path in tqdm(self.used_subscenes, desc="Loading poses"):
                    # subscene_path: /root/.../scene0000_00/00000.pkl
                    scene_name = os.path.basename(os.path.dirname(subscene_path))
                    frame_idx = os.path.basename(subscene_path).split('.')[0]
                    with open(subscene_path, 'rb') as f:
                        data = pickle.load(f)
                        pose = data['cam_pose']

                    if scene_name not in self.all_poses:
                        self.all_poses[scene_name] = []
                    self.all_poses[scene_name].append((frame_idx, pose))

                with open(cache_path, 'wb') as f:
                    pickle.dump(self.all_poses, f)

    def __len__(self):
        return len(self.used_subscenes)

    def __getitem__(self, index):
        name = self.used_subscenes[index]
        with open(name, 'rb') as f:
            data = pickle.load(f)

        name_without_ext = os.path.splitext(name)[0]
        this_name = name_without_ext.split('gathered_data/')[-1]

        meta = {}
        meta['name'] = this_name # 'scene0000_00/00000'
        meta['scene_size'] = self.scene_size
        cam_pose = data['cam_pose']
        meta['cam2world'] = cam_pose
        world2cam = np.linalg.inv(cam_pose)
        meta['world2cam'] = world2cam
        
        if self.load_neighbor:
            scene_name = this_name.split('/')[0]
            
            # Find nearest frames in the same scene, except self
            frame_idx = this_name.split('/')[1]
            neighbor_poses = self.all_poses[scene_name]
            # Sort by frame index and exclude self

            # Sort neighbor_poses by pose distance to current frame (excluding self)
            current_pose = cam_pose
            def pose_distance(pose1, pose2):
                # Compute translation distance
                t1 = pose1[:3, 3]
                t2 = pose2[:3, 3]
                return np.linalg.norm(t1 - t2)

            neighbor_poses_sorted = sorted(
                neighbor_poses,
                key=lambda x: pose_distance(current_pose, x[1])
            )

            nearest_neighbors = []
            for idx, (nbr_idx, nbr_pose) in enumerate(neighbor_poses_sorted):
                if nbr_idx != frame_idx:
                    nearest_neighbors.append((nbr_idx, nbr_pose))
                if len(nearest_neighbors) >= self.load_neighbor_random_all:
                    break

            if len(nearest_neighbors) != self.load_neighbor_random_all:
                return self[np.random.randint(0, len(self))]

            nearest_neighbors = [nearest_neighbors[i] for i in np.random.choice(self.load_neighbor_random_all, self.load_neighbor, replace=False)]

            meta['neighbor_frames'] = [os.path.join(f'{self.occscannet_root}/posed_images/', f'{scene_name}/{nbr[0]}.jpg') for nbr in nearest_neighbors]
            meta['neighbor_rgbs'] = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) / 255.0 for p in meta['neighbor_frames']]
            meta['neighbor_cam2world'] = [nbr[1] for nbr in nearest_neighbors]

        rgb_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.jpg'
        depth_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.png'
        depth_gt_np = Image.open(depth_path).convert('I;16')
        depth_gt_np = np.array(depth_gt_np) / 1000.0

        if self.text_path:
            text_path = f'{self.text_path}/{this_name}.txt'
            if not os.path.exists(text_path):
                print(f"Text file {text_path} does not exist, skipping sample.")
                return self[np.random.randint(0, len(self))]
            with open(text_path, 'r') as f:
                text = f.read()
            classnames = text.strip().split(',')
            classnames = [t.strip().lower() for t in classnames]
            classnames = np.unique(classnames).tolist()
            meta['classnames'] = classnames

        transform = Compose([
            Resize(
                width=480,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        img_depthbranch = cv2.imread(rgb_path)
        img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
        img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
        sample = transform({
            'image': img_depthbranch,
            'depth': depth_gt_np
        })
        img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
        depth_gt_np = torch.from_numpy(sample['depth']).unsqueeze(0)
        meta['depth_gt_np'] = depth_gt_np
        depth_valid_mask = (torch.isnan(depth_gt_np) == 0)
        depth_gt_np[depth_valid_mask == 0] = 0
        meta['img_depthbranch'] = img_depthbranch
        meta['depth_gt_np_valid'] = depth_gt_np
        meta['rgb_path'] = rgb_path
        N_img = []
        this_img = imread(rgb_path, 'unchanged').astype(np.float32)
        this_H, this_W, _ = this_img.shape

        # resize
        if not self.vggt_image_preprocess:
            new_H, new_W = self.new_H, self.new_W
        else:
            target_size = self.target_size
            if this_W >= this_H:
                new_width = target_size
                new_height = round(this_H * (new_width / this_W) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(this_W * (new_height / this_H) / 14) * 14  
            new_W, new_H = new_width, new_height
            assert new_height < target_size, 'need crop, not implemented for now'

        new_img = cv2.resize(this_img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
        W_factor = new_W / this_W
        H_factor = new_H / this_H
        this_H, this_W = new_H, new_W

        N_img.append(new_img)
        img = np.stack(N_img, 0) # [1, 968, 1296, 3]
        img = [img] # [1, 1, 968, 1296, 3]

        cam_intrin = data['intrinsic']
        cam_intrin[0, 0] *= W_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[1, 2] *= H_factor

        meta['cam_k'] = cam_intrin[:3, :3]
        viewpad = np.eye(4)
        viewpad[:meta['cam_k'].shape[0], :meta['cam_k'].shape[1]] = meta['cam_k']
        meta['cam2img'] = viewpad
        world2img = (viewpad @ world2cam)
        meta['world2img'] = world2img
        meta['depth_path'] = depth_path
        depth_gt = Image.open(depth_path).convert('I;16')
        depth_gt = np.array(depth_gt) / 1000.0
        meta['depth_gt'] = depth_gt

        vox_origin = data["voxel_origin"]

        meta['vox_origin'] = np.round(np.array(vox_origin, dtype=np.float32), 4)
        target = data["target_1_4"] # 60, 60, 36
        target = np.transpose(target, (1, 0, 2))
        # 把代表unknown的255换成0，把代表空的0换成12
        target[target == 0] = 12
        target[target == 255] = 0 
        occ = target # (60, 60, 36)
        nonemptymask = (occ != 12)
        occ = [occ] # [1, 60, 60, 36]

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z, occ_xyz = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=True,
        )

        _, fov_mask_4, _, _ = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size * 4,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=False,
        )
        meta['projected_pix'] = projected_pix
        meta['fov_mask'] = fov_mask.reshape(60, 60, 36)
        meta['fov_mask_4'] = fov_mask_4.reshape(15, 15, 9)

        meta['pix_z'] = pix_z
        meta['occ_xyz'] = occ_xyz.reshape(60, 60, 36, 3)

        vox_near = meta['vox_origin']
        vox_far = vox_near + meta['scene_size']
        nyu_pc_range = np.concatenate([vox_near, vox_far], axis=0)
        meta['nyu_pc_range'] = nyu_pc_range
        
        scan = meta['occ_xyz'][nonemptymask]
        meta['occ_xyz_nonempty'] = scan
        meta['num_depth'] = self.num_pts
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.01
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], nyu_pc_range[0], nyu_pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], nyu_pc_range[1], nyu_pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], nyu_pc_range[2], nyu_pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - nyu_pc_range[0]) / (nyu_pc_range[3] - nyu_pc_range[0])
        scan[:, 1] = (scan[:, 1] - nyu_pc_range[1]) / (nyu_pc_range[4] - nyu_pc_range[1])
        scan[:, 2] = (scan[:, 2] - nyu_pc_range[2]) / (nyu_pc_range[5] - nyu_pc_range[2])

        meta['anchor_points'] = scan

        cam_vox_near = np.array([-5, -6, -3])
        cam_vox_far = np.array([5, 6, 8])
        cam_vox_range = np.concatenate([cam_vox_near, cam_vox_far], axis=0).astype(np.float32)
        meta['cam_vox_range'] = cam_vox_range

        meta['occ_mask_valid'] = (occ != 0)
        meta['occ_mask_valid_fov'] = (occ != 0) & fov_mask
        meta['label'] = occ

        imgs = np.stack(img, 0)
        occs = np.stack(occ, 0)
        data_tuple = (imgs, meta, occs)
        return data_tuple

    def get_meshgrid(self, ranges, grid, reso):
        pass
    
    def get_data_info(self, info):
        pass

    def get_scene_index(self, scene_name=None):
        pass



def read_depth(depth_path):
    depth_vis = Image.open(depth_path).convert('I;16')  
    depth_vis_array = np.array(depth_vis)

    arr1 = np.right_shift(depth_vis_array, 3)
    arr2 = np.left_shift(depth_vis_array, 13)
    depth_vis_array = np.bitwise_or(arr1, arr2)

    depth_inpaint = depth_vis_array.astype(np.float32) / 1000.0  
      
    return depth_inpaint

