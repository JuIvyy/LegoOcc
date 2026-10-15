"""Single-image LegoOcc demo: image -> open-vocabulary occupancy -> colored voxel point cloud (.ply).

    python demo.py --model JuIvyy/LegoOcc --image examples/scene0000_00_00061.jpg
    python demo.py --model JuIvyy/LegoOcc --image examples/scene0381_00_00009.jpg --vocab "armchair,round table,floor,wall,window"
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch

from legoocc_hub import LegoOcc

# RGB colors for the default vocabulary (index 0 = empty, never written);
# custom vocabularies cycle through EXTRA_COLORS.
PALETTE = np.array([
    [255, 255, 255],  # empty
    [22, 191, 206],   # ceiling
    [214, 38, 40],    # floor
    [43, 160, 43],    # wall
    [158, 216, 229],  # window
    [114, 158, 206],  # chair
    [204, 204, 91],   # bed
    [255, 186, 119],  # sofa
    [147, 102, 188],  # table
    [30, 119, 181],   # tvs
    [188, 188, 33],   # furniture
    [255, 127, 12],   # objects
], dtype=np.uint8)
EXTRA_COLORS = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [128, 0, 0], [170, 255, 195],
    [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128], [0, 0, 0],
], dtype=np.uint8)


def save_ply(path, points, colors):
    vertex = np.empty(len(points), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'], vertex['y'], vertex['z'] = points.T
    vertex['red'], vertex['green'], vertex['blue'] = colors.T
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(points)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='JuIvyy/LegoOcc', help='local folder or Hugging Face repo id')
    parser.add_argument('--image', required=True)
    parser.add_argument('--vocab', default=None,
                        help='comma-separated class prompts, e.g. "chair,lamp,potted plant" '
                             '(default: the 11 OccScanNet classes; custom ones need open_clip_torch)')
    parser.add_argument('--camera', default=None,
                        help='JSON with "intrinsics" (3x3) and optionally "cam2world" (4x4) / "vox_origin" (3); '
                             'defaults to <image>.json next to the image if present')
    parser.add_argument('--pitch', type=float, default=25.0, help='camera tilt (deg) when cam2world is unknown')
    parser.add_argument('--cam-height', type=float, default=1.5, help='camera height (m) when cam2world is unknown')
    parser.add_argument('--fov-only', action='store_true', help='only keep voxels visible in the image')
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()

    camera = {}
    cam_file = args.camera or osp.splitext(args.image)[0] + '.json'
    if osp.exists(cam_file):
        with open(cam_file) as f:
            camera = json.load(f)
        print(f'Using camera parameters from {cam_file}: {sorted(camera)}')
    else:
        print('No camera file found: using default intrinsics and camera pose.')

    model = LegoOcc.from_pretrained(args.model).cuda().eval()
    out = model.predict(
        args.image,
        intrinsics=camera.get('intrinsics'),
        cam2world=camera.get('cam2world'),
        vox_origin=camera.get('vox_origin'),
        vocabulary=[v.strip() for v in args.vocab.split(',')] if args.vocab else None,
        pitch_deg=args.pitch,
        cam_height=args.cam_height,
    )

    occ = out['occupancy'].cpu().numpy()                  # (60, 60, 36), 0 = empty
    centers = out['voxel_centers'].cpu().numpy()          # (60, 60, 36, 3), world frame
    names = out['class_names']                            # ['empty', class 1, class 2, ...]
    palette = PALETTE if args.vocab is None else np.concatenate(
        [PALETTE[:1], np.resize(EXTRA_COLORS, (len(names) - 1, 3))])
    keep = occ != 0
    if args.fov_only:
        keep &= out['fov_mask'].cpu().numpy()

    os.makedirs(args.out, exist_ok=True)
    stem = osp.splitext(osp.basename(args.image))[0]
    ply_path = osp.join(args.out, f'{stem}_occ.ply')
    save_ply(ply_path, centers[keep], palette[occ[keep]])

    print(f'Saved {keep.sum()} occupied voxels to {ply_path}')
    for c in range(1, len(names)):
        n = int((occ[keep] == c).sum())
        if n:
            print(f'  {names[c]:<14s} {n:6d}  color {tuple(palette[c].tolist())}')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
