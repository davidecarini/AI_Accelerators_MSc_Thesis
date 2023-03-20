#!/bin/bash

# Author: Davide Carini

echo "-----------------------------------------"
echo "MAKE TARGET U280 STARTED.."
echo "-----------------------------------------"

TARGET_U280=./target_u280
COMPILE_U280=./compiled_model/U280
APP=./application
NET_NAME=customcnn

# remove previous results
rm -rf ${TARGET_U280}
mkdir -p ${TARGET_U280}/model_dir

# copy application to TARGET_U50 folder
cp ${APP}/*.py ${TARGET_U280}
echo "  Copied application to TARGET_U280 folder"


# copy xmodel to TARGET_U50 folder
cp ${COMPILE_U280}/${NET_NAME}.xmodel ${TARGET_U280}/model_dir/.
echo "  Copied xmodel file(s) to TARGET_U280 folder"

# create image files and copy to target folder
mkdir -p ${TARGET_U280}/images

python generate_images.py  \
    --dataset=mnist \
    --image_dir=${TARGET_U280}/images \
    --image_format=jpg \
    --max_images=100000

echo "  Copied images to TARGET_U280 folder"

echo "-----------------------------------------"
echo "MAKE TARGET U280 COMPLETED"
echo "MAKE TARGET U280 COMPLETED"
echo "-------------------------------------u280"
