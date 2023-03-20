#!/bin/bash


# Author: Davide Carini

ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json
OUTDIR=./compiled_model/U280
NET_NAME=customcnn
MODEL=./models/quantized_model.h5


compile() {
  vai_c_tensorflow2 \
    --model      $MODEL\
    --arch       $ARCH \
    --output_dir $OUTDIR \
    --net_name   $NET_NAME
}

echo "-----------------------------------------"
echo "COMPILE U280 STARTED.."
echo "-----------------------------------------"

compile 2>&1 | tee compile.log

echo "-----------------------------------------"
echo "COMPILE U280 COMPLETED"
echo "-----------------------------------------"
~                                                        
