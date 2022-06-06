#!/bin/bash

# This script copies the 100 randomly selected classes, into a new ImageNet partition directory.
# The 100 classes are specified in the provided .txt file.
# The new partition will be contained within the ImageNet root directory.

# TODO <------------ Specify ImageNet root directory
IMNET_ROOT_DIR=unspecified

# TODO <------------- Specify path to the '100_rnd_cls_list.txt' file contained in code_root/imnet_100
WANTED_CLS_LST=unspecified

# Make directories for the new partition
mkdir ${IMNET_ROOT_DIR}/2012-100
mkdir ${IMNET_ROOT_DIR}/2012-100/val
mkdir ${IMNET_ROOT_DIR}/2012-100/train

# Copy specified classes to the new partition
while read line; do
    cp -r ${IMNET_ROOT_DIR}/2012-1k/train/${line} ${IMNET_ROOT_DIR}/2012-100/train/
    cp -r ${IMNET_ROOT_DIR}/2012-1k/val/${line} ${IMNET_ROOT_DIR}/2012-100/val/
done < $WANTED_CLS_LST
