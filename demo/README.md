# LegoOcc Demo

Single-image **open-vocabulary** occupancy prediction with LegoOcc. Give it one RGB image and it predicts a 60×60×36 occupancy grid (8 cm voxels, a 4.8 m × 4.8 m × 2.88 m box) and labels each occupied voxel with a class from a vocabulary you choose. The result is saved as a colored point cloud of voxel centers (`.ply`).

This folder is self-contained. It only needs PyTorch and a few small packages: no mmcv/mmdet/mmengine, SAM, DINO, pytorch3d, or gsplat. The CUDA op is compiled automatically the first time you load the model.

## Installation

```bash
cd demo
pip install -r requirements.txt
pip install open_clip_torch      # optional, only needed for custom vocabularies
```

You also need PyTorch with CUDA and the CUDA toolkit (`nvcc`). The toolkit version must match the CUDA version of your PyTorch build. If `nvcc` is not on your `PATH`, set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
```

The first time the model is loaded, the Gaussian-to-voxel CUDA op is compiled (about 1–2 min). The result is cached in `~/.cache/torch_extensions`, so later runs start immediately.

## Model

| Repo | Backbone | Params | Size |
|---|---|---|---|
| `JuIvyy/LegoOcc` | Depth-Anything-V2 ViT-B | 100M | 0.4 GB |

The checkpoint already contains the CLIP text features of the default vocabulary, which is the 11 OccScanNet classes: ceiling, floor, wall, window, chair, bed, sofa, table, television, furniture, objects. The default vocabulary therefore works without CLIP installed.

## Quick Start

The example images in `examples/` come from the ScanNet test split. Each image has a JSON file with its camera parameters:

```bash
# default vocabulary
python demo.py --model JuIvyy/LegoOcc --image examples/scene0000_00_00061.jpg

# your own vocabulary (needs open_clip_torch)
python demo.py --model JuIvyy/LegoOcc --image examples/scene0472_00_00061.jpg \
    --vocab "sink,stove,cabinet,countertop,floor,wall"
```

Outputs:

- `outputs/<name>_occ.ply`: occupied voxel centers, colored by class. The demo prints which color goes with which class.

> **About the output.** The `.ply` file is **not** a reconstructed mesh, a depth point cloud, or the predicted Gaussians. It contains **one point per occupied voxel**, placed at the **voxel center**. Empty voxels are not saved, so the points form a regular 3D grid with 8 cm spacing. Each point is colored by its predicted class (the closest prompt in your vocabulary). Coordinates are in meters, in the world frame given by `cam2world` (z up).

Open the `.ply` file in MeshLab, CloudCompare, or Open3D.

## Using Your Own Images

```bash
python demo.py --model JuIvyy/LegoOcc --image my_room.jpg --vocab "desk,office chair,monitor,floor,wall"
```

Camera parameters are read from `--camera cam.json`. If that is not given, the demo looks for `<image>.json` next to the image. Every field is optional:

```json
{
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "cam2world": [[...4x4...]],
  "vox_origin": [x, y, z]
}
```

- `intrinsics`: the camera matrix at the original image resolution. If missing, the demo assumes a ScanNet-like field of view (`fx = fy = 0.9 * width`). Use real intrinsics when you have them, because depth scale depends on them.
- `cam2world`: the camera pose in a gravity-aligned, z-up world frame. It only decides where the output grid is placed and how it is oriented. If missing, the camera is placed `--cam-height` meters (default 1.5) above the floor and tilted down by `--pitch` degrees (default 25, the median tilt in ScanNet; use `--pitch 0` for level photos).
- `vox_origin`: the world position of the grid corner. If missing, the 4.8 m box is placed the way OccScanNet does it: the bottom sits on the floor and the box extends forward from the camera.

Add `--fov-only` to keep only the voxels that are visible in the image.

**Vocabulary tips:**

- Every occupied voxel is assigned to the closest prompt, so include the background classes (`floor`, `wall`, ...) in your vocabulary.
- Short noun phrases work best, e.g. `"potted plant"`, `"office chair"`.

## Python API

```python
from legoocc_hub import LegoOcc

model = LegoOcc.from_pretrained("JuIvyy/LegoOcc").cuda().eval()
out = model.predict("examples/scene0000_00_00061.jpg", intrinsics=K)          # default vocabulary
out = model.predict("room.jpg", vocabulary=["sofa", "lamp", "floor", "wall"])   # one-off vocabulary
model.set_vocabulary(["sofa", "lamp", "floor", "wall"])                         # or change it permanently

out["occupancy"]      # (60, 60, 36) labels: 0 = empty, k = out["class_names"][k]
out["class_names"]    # ["empty", "sofa", "lamp", ...]
out["similarity"]     # (K, 60, 60, 36) cosine similarity to each prompt
out["empty_prob"]     # (60, 60, 36) probability of free space
out["voxel_centers"]  # (60, 60, 36, 3) world coordinates
out["gaussians"]      # language-embedded 3D Gaussians (camera frame)
```

Pass `return_features=True` to also get the `(512, 60, 60, 36)` voxel language features. You can use them to run any text query later.

## License

The model weights are derived from Depth-Anything-V2-Base and are released for non-commercial research use only. The Gaussian aggregation CUDA op builds on Inria's GRAPHDECO code (non-commercial license). The example images are from [ScanNet](http://www.scan-net.org/) and are subject to the ScanNet Terms of Use.
