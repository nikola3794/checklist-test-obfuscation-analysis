#!/bin/bash

# Specify directories
CODE_ROOT_DIR=unspecified
DATA_SET_ROOT_DIR=unspecified
CHECKPOINT_FILE_PATH=unspecified

# Specify model
MODEL=resnet_50_PNI_W_layerwise
# Specify number of classes
NUM_CLASSES=100
# Specify batch size
BATCH_SIZE_REG=128
BATCH_SIZE_AE=10
# SPecify number of workers
NUM_WORKERS=8

#Specify adversarial parameters
AE_EPS=4.0 # 2.0 or 4.0 used
AE_PGD_STEP=1.0 # eps=2-->step=1.0 ... eps=4--->step=1.0
AE_PGD_N_STEPS=10
AE_N_EOT_SAMPLES=25
AE_N_REPEATS=5
AE_RND_INIT=True

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

# Evaluate regular
python3 validate_regular.py \
--data_dir=${DATA_SET_ROOT_DIR} \
--checkpoint=${CHECKPOINT_ROOT_DIR} \
--num-classes=${NUM_CLASSES} \
--model=${MODEL} \
--workers=${NUM_WORKERS} \
--batch-size=${BATCH_SIZE_REG}

# Evaluate FSGM
python3 validate_robust.py \
--data_dir=${DATA_SET_ROOT_DIR} \
--checkpoint=${CHECKPOINT_ROOT_DIR} \
--results-file=${CHECKPOINT_ROOT_DIR} \
--num-classes=${NUM_CLASSES} \
--model=${MODEL} \
--workers=${NUM_WORKERS} \
--batch-size=${BATCH_SIZE_AE} \
--ae-eps-list ${AE_EPS} \
--ae-step-list ${AE_EPS} \
--ae-n-steps=1 \
--ae-n-expectation-samples 1 \
--ae-n-restarts=${AE_N_REPEATS}

# Evaluate FSGM (with EoT-25)
python3 validate_robust.py \
--data_dir=${DATA_SET_ROOT_DIR} \
--checkpoint=${CHECKPOINT_ROOT_DIR} \
--results-file=${CHECKPOINT_ROOT_DIR} \
--num-classes=${NUM_CLASSES} \
--model=${MODEL} \
--workers=${NUM_WORKERS} \
--batch-size=${BATCH_SIZE_AE} \
--ae-eps-list ${AE_EPS} \
--ae-step-list ${AE_EPS} \
--ae-n-steps=1 \
--ae-n-expectation-samples ${AE_N_EOT_SAMPLES} \
--ae-n-restarts=${AE_N_REPEATS}

#Evaluate PGD-10
python3 validate_robust.py \
--data_dir=${DATA_SET_ROOT_DIR} \
--checkpoint=${CHECKPOINT_ROOT_DIR} \
--results-file=${CHECKPOINT_ROOT_DIR} \
--num-classes=${NUM_CLASSES} \
--model=${MODEL} \
--workers=${NUM_WORKERS} \
--batch-size=${BATCH_SIZE_AE} \
--ae-eps-list ${AE_EPS} \
--ae-step-list ${AE_PGD_STEP} \
--ae-n-steps=${AE_PGD_N_STEPS} \
--ae-n-expectation-samples 1 \
--ae-n-restarts=${AE_N_REPEATS}

#Evaluate PGD-10 (with EoT-25)
python3 validate_robust.py \
--data_dir=${DATA_SET_ROOT_DIR} \
--checkpoint=${CHECKPOINT_ROOT_DIR} \
--results-file=${CHECKPOINT_ROOT_DIR} \
--num-classes=${NUM_CLASSES} \
--model=${MODEL} \
--workers=${NUM_WORKERS} \
--batch-size=${BATCH_SIZE_AE} \
--ae-eps-list ${AE_EPS} \
--ae-step-list ${AE_PGD_STEP} \
--ae-n-steps=${AE_PGD_N_STEPS} \
--ae-n-expectation-samples ${AE_N_EOT_SAMPLES} \
--ae-n-restarts=${AE_N_REPEATS}