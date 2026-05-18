# Training and Evaluation

All scripts should be run from the project root unless noted otherwise.

## Training

Train LegoOcc with:

```bash
cd scripts
bash train.sh <NUM_GPUS> <CONFIG>
```

Example:

```bash
cd scripts
bash train.sh 4 legoocc_release
```

This launches `torchrun` and writes outputs to:

```text
work_dirs/<CONFIG>/
```

## Evaluation

```bash
cd scripts
bash eval.sh 4 legoocc_release
```
