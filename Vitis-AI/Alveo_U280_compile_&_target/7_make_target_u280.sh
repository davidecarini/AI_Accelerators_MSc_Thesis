#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey


echo "-----------------------------------------"
echo "MAKE TARGET U280 STARTED.."
echo "-----------------------------------------"

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
echo "-----------------------------------------"




