# Installation

## Prerequisites

- Python 3.11
- CUDA 12.1
- GCC >= 7.5

## 1. Clone the Repository

```bash
git clone https://github.com/xxx/LegoOcc.git
cd LegoOcc
```

## 2. Create a Conda Environment

```bash
conda create -n legoocc python=3.11 -y
conda activate legoocc
```

## 3. Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install MM-Series Libraries

```bash
pip install openmim
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmsegmentation==1.2.2
mim install mmdet3d==1.4.0
```

## 5. Install Additional Dependencies


```bash
conda install pytorch3d -c pytorch3d

pip install tqdm einops open3d timm pyquaternion
```


## 5. Build Custom CUDA Operators

```bash
cd src/legoocc/model/head/gaussian_occ_head/ops/localagg_prob
python setup.py build_ext --inplace
cd -
```

