#!/bin/bash

# Specify directories
CODE_ROOT_DIR=unspecified
DATA_SET_ROOT_DIR=unspecified
OUTPUT_ROOT_DIR=unspecified

#Specify model
MODEL=vit_deit_small_patch16_224_PNI_W_elementwise
# Specify number of classes
NUM_CLASSES=1000
#Specify adversarial parameters
AE_EPS=4.0 # 2.0 or 4.0 used
AE_FGSM_STEP=5.0 # eps=2-->step=2.5 ... eps=4--->step=5.0
AE_N_REPEATS=1
AE_RND_INIT=True

#Other parameters (mostly unchanged between experiments)
CONFIG_ROOT_DIR=${CODE_ROOT_DIR}/experiment_configs/default_training_config_imnet_transfer.yaml
N_GPU=2

# Activate appropriate python environment
# TODO <-----------------

# Print hardware stats
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Number of CPU threads/core: $(nproc --all)"

# Print dataset stats
echo "Number of train classes: $(ls ${DATA_SET_ROOT_DIR}/train | wc -l)"
echo "Number of validation classes: $(ls ${DATA_SET_ROOT_DIR}/val | wc -l)"

# Set project paths
export PYTHONPATH=${PYTHONPATH}:${CODE_ROOT_DIR}
cd ${CODE_ROOT_DIR}
pwd

# Quick fix to a problem that was occuring when I ran multiple jobs on the same GPU node
RND=$(( RANDOM % 999 ))
MASTER_PORT=$((29000 + $RND))
echo "Master port: ${MASTER_PORT}"

python3 -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${MASTER_PORT} train_robust.py \
--config=${CONFIG_ROOT_DIR} \
--data_dir=${DATA_SET_ROOT_DIR} \
--output=${OUTPUT_ROOT_DIR} \
--num-classes=${NUM_CLASSES} \
--model=${MODEL} \
--ae-clip-eps=${AE_EPS} \
--ae-fgsm-step=${AE_FGSM_STEP} \
--ae-n-repeats=${AE_N_REPEATS} \
--ae-random-init=${AE_RND_INIT} \

