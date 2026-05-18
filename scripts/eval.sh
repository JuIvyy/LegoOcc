export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONUTF8=1

export NCCL_P2P_DISABLE=1

GPUS=$1
CFG=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


PYTHONPATH=`pwd`/src/:`pwd`/src/legoocc/Depth-Anything-V2/metric_depth:`pwd`/src/legoocc/Trident:$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    scripts/train_mono.py \
        --py-config config/${CFG}.py \
        --work-dir work_dirs/${CFG} \
        --evaluate
