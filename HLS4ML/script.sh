#!/bin/bash

XRT=/opt/xilinx/xrt/setup.sh
VITIS=/opt/xilinx/Vitis/2020.1/settings64.sh
VIVADO=/opt/xilinx/Vivado/2020.1/settings64.sh
PY=/usr/local/bin/python3.7
PIP=/usr/local/bin/pip3.7
ENV=./tf_env

if [ ! -d "$ENV" ]
then
  ${PY} -m venv $ENV
  source $ENV/bin/activate
  ${PIP} install -U pip
  ${PIP} install -U setuptools
  ${PIP} install -r requirements.txt
  source $XRT
  source $VITIS
  source $VIVADO
else
  source $ENV/bin/activate
  source $XRT
  source $VITIS
  source $VIVADO
fi


${PY} test.py
