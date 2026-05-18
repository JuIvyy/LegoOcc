# Data Preparation

## OccScanNet

LegoOcc is trained and evaluated on **OccScanNet** for monocular open-vocabulary occupancy prediction in indoor scenes.

## Directory Structure

After preparation, the `data/` directory should look like:

```text
data/
└── occscannet/
    ├── train_final.txt
    ├── test_final.txt
    ├── train_mini_final.txt
    ├── test_mini_final.txt
    ├── gathered_data -> /path/to/OccScanNet/gathered_data
    ├── posed_images -> /path/to/OccScanNet/posed_images
    └── qwen25vl_7b_objects -> /path/to/text_annotations
```

## OccScanNet

1. Download the dataset from [hongxiaoy/OccScanNet](https://huggingface.co/datasets/hongxiaoy/OccScanNet).
2. Unzip the downloaded files.
3. Create symbolic links under `data/occscannet/`:

```bash
cd data/occscannet
ln -s /path/to/OccScanNet/gathered_data
ln -s /path/to/OccScanNet/posed_images
cd ../..
```
