#!/bin/bash
#
################################################################################################
# Script taken from: https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md #
################################################################################################
#
#                               ###################
#                               #  Instructions:  #
#                               ###################
#
# Script to extract the ImageNet dataset from downloaded .tar files.
# The .tar files can be downloaded at: http://image-net.org/download-images 
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
# Make sure that ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar are in your current directory.
# (the current directory should be the data set root directory)
#
# After the extraction, the directory structure will look like this:
# root
#  ├──train/
#  │    ├── n01440764
#  │    │   ├── n01440764_10026.JPEG
#  │    │   ├── n01440764_10027.JPEG
#  │    │   ├── ......
#  │    ├── ......
#  ├──val/
#  │    ├── n01440764
#  │    │   ├── ILSVRC2012_val_00000293.JPEG
#  │    │   ├── ILSVRC2012_val_00002138.JPEG
#  │    │   ├── ......
#  │    ├── ......

# Extract the training data:
#
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && mv ILSVRC2012_img_train.tar ../ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

#
# Extract the validation data and move images to subfolders:
#
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash



# Check total files after extract
find train/ -name "*.JPEG" | wc -l
#  Should be 1281167
find val/ -name "*.JPEG" | wc -l
#  Should be 50000