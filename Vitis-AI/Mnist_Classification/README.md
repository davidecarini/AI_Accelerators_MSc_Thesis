# Vitis-AI 1.4 (Mnist-classification TensorFlow2)
Vitis-AI 1.4 TensorFlow2 flow with a custom CNN model, targeted ALveo U280 evaluation board.

## Introduction
The project should be run in Vitis-AI 1.4 Docker, vitis-ai-tensorflow2 conda environment.Follow https://github.com/Xilinx/Vitis-AI to setup the environment before starting.

This tutorial is quite similar to the Xilinx released tutorial https://github.com/Xilinx/Vitis-AI-Tutorials/tree/MNIST-Classification-TensorFlow.

Refer to http://yann.lecun.com/exdb/mnist/ for the Mnist hand-written digits dataset.

We will run the following steps:

* Training and evaluation of a small custom convolutional neural network using TensorFlow2.
* Quantization of the floating-point model.
* Evaluation of the quantized model.
* Apply finetuning to the trained model with a calibration dataset.
* Compilation of both the quantized & finetuned model to create the .xmodel files ready for execution on the DPU accelerator IP.
* Download and run the application on the U280 evaluation board.

## Python scripts
*load_data.py*: 
load Mnist dataset

*generate_images.py*: 
Generate local images from Keras datasets. This file is form https://github.com/Xilinx/Vitis-AI-Tutorials/tree/MNIST-Classification-TensorFlow 

*train.py*: 
Create & train a simple CNN model for Mnist classification. A trained floating point model will be saved.

*quantize.py*: 
Quantize the saved floating point model with Vitis Quantizer. A quantized model will be saved.

*eval_quantized.py*: 
Evaluate the quantized model.

*finetune.py*: 
Model finetuning.

## Shell scripts
*compile_u280.sh*: 
Launches the vai_c_tensorflow2 command to compile the quantized or finetuned model into an .xmodel file for the U280 evaluation board

*make_target_u280.sh*: 
Copies the .xmodel and images to the ./target_u280 folder ready to be copied to the U280 evaluation board's SD card.

## Implement
Before running this part, we should setup Vitis-AI docker and activate vitis-ai-tensorflow2 anaconda environment.
For more details, refer to the latest version of the Vitis AI User Guide (UG1414). 

### Set the environment
```
Vitis-AI /workspace/test/src > source setenv.sh

```


### Build and train model

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src > python train.py 


Load Mnist dataset..

Create custom cnn..
Model: "mnist_customcnn_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 12)        120       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 12)        0         
_________________________________________________________________
flatten (Flatten)            (None, 2028)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                20290     
=================================================================
Total params: 20,410
Trainable params: 20,410
Non-trainable params: 0
_________________________________________________________________

Fit on dataset..
Epoch 1/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4035 - accuracy: 0.8895 - val_loss: 0.1952 - val_accuracy: 0.9459
Epoch 2/10
782/782 [==============================] - 5s 6ms/step - loss: 0.1764 - accuracy: 0.9496 - val_loss: 0.1303 - val_accuracy: 0.9663
Epoch 3/10
782/782 [==============================] - 5s 6ms/step - loss: 0.1241 - accuracy: 0.9648 - val_loss: 0.1047 - val_accuracy: 0.9719
Epoch 4/10
782/782 [==============================] - 5s 6ms/step - loss: 0.0992 - accuracy: 0.9715 - val_loss: 0.0911 - val_accuracy: 0.9745
Epoch 5/10
782/782 [==============================] - 5s 6ms/step - loss: 0.0843 - accuracy: 0.9759 - val_loss: 0.0827 - val_accuracy: 0.9764
Epoch 6/10
782/782 [==============================] - 5s 6ms/step - loss: 0.0738 - accuracy: 0.9786 - val_loss: 0.0771 - val_accuracy: 0.9781
Epoch 7/10
782/782 [==============================] - 5s 6ms/step - loss: 0.0659 - accuracy: 0.9809 - val_loss: 0.0733 - val_accuracy: 0.9793
Epoch 8/10
782/782 [==============================] - 5s 6ms/step - loss: 0.0595 - accuracy: 0.9824 - val_loss: 0.0707 - val_accuracy: 0.9797
Epoch 9/10
782/782 [==============================] - 5s 6ms/step - loss: 0.0542 - accuracy: 0.9843 - val_loss: 0.0691 - val_accuracy: 0.9794
Epoch 10/10
782/782 [==============================] - 5s 6ms/step - loss: 0.0498 - accuracy: 0.9857 - val_loss: 0.0681 - val_accuracy: 0.9797

Save trained model to./models/float_model.h5.

Evaluate model on test dataset..
157/157 [==============================] - 0s 3ms/step - loss: 0.0598 - accuracy: 0.9810
loss: 0.060
acc: 0.981

```

### Quantize the floating-point model

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src > python quantize.py 
Load float model..

Load Mnist dataset..

Run quantization..
[VAI INFO] Start CrossLayerEqualization...
10/10 [==============================] - 0s 26ms/step
[VAI INFO] CrossLayerEqualization Done.
[VAI INFO] Start Quantize Calibration...
157/157 [==============================] - 2s 14ms/step
[VAI INFO] Quantize Calibration Done.
[VAI INFO] Start Post-Quantize Adjustment...
[VAI INFO] Post-Quantize Adjustment Done.
[VAI INFO] Quantization Finished.

Saved quantized model as ./models/quantized_model.h5

```

### Evaluate quantized model

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > python eval_quantized.py 
Load float model..

Load quantized model..

Load Mnist dataset..

Compile model..

Evaluate model on test Dataset
157/157 [==============================] - 1s 4ms/step - loss: 0.0596 - accuracy: 0.9811
loss: 0.060
acc: 0.981


```
### Finetuning

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src > python finetune.py 
Load Mnist dataset..

Create quantize training model..
[VAI INFO] Start CrossLayerEqualization...
10/10 [==============================] - 0s 27ms/step
[VAI INFO] CrossLayerEqualization Done.
[VAI INFO] Start Generation of Quantize-aware Training Model.
[VAI INFO] Generation of Quantize-aware Training Model Done.

Compiling model..

Fit on Dataset..
Epoch 1/10
782/782 [==============================] - 10s 13ms/step - loss: 0.0449 - accuracy: 0.9867 - val_loss: 0.0652 - val_accuracy: 0.9807
Epoch 2/10
782/782 [==============================] - 10s 13ms/step - loss: 0.0432 - accuracy: 0.9875 - val_loss: 0.0643 - val_accuracy: 0.9820
Epoch 3/10
782/782 [==============================] - 11s 13ms/step - loss: 0.0418 - accuracy: 0.9880 - val_loss: 0.0642 - val_accuracy: 0.9821
Epoch 4/10
782/782 [==============================] - 11s 14ms/step - loss: 0.0407 - accuracy: 0.9884 - val_loss: 0.0646 - val_accuracy: 0.9823
Epoch 5/10
782/782 [==============================] - 10s 12ms/step - loss: 0.0398 - accuracy: 0.9888 - val_loss: 0.0651 - val_accuracy: 0.9821
Epoch 6/10
782/782 [==============================] - 10s 12ms/step - loss: 0.0385 - accuracy: 0.9892 - val_loss: 0.0650 - val_accuracy: 0.9819
Epoch 7/10
782/782 [==============================] - 9s 12ms/step - loss: 0.0381 - accuracy: 0.9894 - val_loss: 0.0652 - val_accuracy: 0.9820
Epoch 8/10
782/782 [==============================] - 11s 13ms/step - loss: 0.0377 - accuracy: 0.9894 - val_loss: 0.0652 - val_accuracy: 0.9822
Epoch 9/10
782/782 [==============================] - 13s 16ms/step - loss: 0.0376 - accuracy: 0.9896 - val_loss: 0.0674 - val_accuracy: 0.9824
Epoch 10/10
782/782 [==============================] - 12s 16ms/step - loss: 0.0366 - accuracy: 0.9898 - val_loss: 0.0679 - val_accuracy: 0.9820

Saved finetuned model as ./models/finetuned_model.h5

Evaluate model on test Dataset..
157/157 [==============================] - 1s 4ms/step - loss: 0.0636 - accuracy: 0.9811
loss: 0.064
acc: 0.981

```

### Compile into DPU model file

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src > bash -x compile_u280.sh 

+ ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json
+ OUTDIR=./compiled_model/U280
+ NET_NAME=customcnn
+ MODEL=./models/quantized_model.h5
+ echo -----------------------------------------
-----------------------------------------
+ echo 'COMPILE U280 STARTED..'
COMPILE U280 STARTED..
+ echo -----------------------------------------
-----------------------------------------
+ compile
+ tee compile.log
+ vai_c_tensorflow2 --model ./models/quantized_model.h5 --arch /opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json -
-output_dir ./compiled_model/U280 --net_name customcnn
[INFO] Namespace(batchsize=1, inputs_shape=None, layout='NHWC', model_files=['./models/quantized_model.h5'], model_ty
pe='tensorflow2', named_inputs_shape=None, out_filename='/tmp/customcnn_org.xmodel', proto=None)
[INFO] tensorflow2 model: /workspace/test/src/models/quantized_model.h5
[INFO] keras version: 2.4.0
[INFO] Tensorflow Keras model type: functional
[INFO] parse raw model     :100%|██████████| 8/8 [00:00<00:00, 16108.70it/s]                 
[INFO] infer shape (NHWC)  :100%|██████████| 13/13 [00:00<00:00, 16374.16it/s]               
[INFO] perform level-0 opt :100%|██████████| 2/2 [00:00<00:00, 777.01it/s]                   
[INFO] perform level-1 opt :100%|██████████| 2/2 [00:00<00:00, 2915.75it/s]                  
[INFO] generate xmodel     :100%|██████████| 13/13 [00:00<00:00, 4478.52it/s]                
[INFO] dump xmodel: /tmp/customcnn_org.xmodel
[UNILOG][INFO] Target architecture: DPUCAHX8H_ISA2
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCAHX8H_ISA2
[UNILOG][INFO] Graph name: mnist_customcnn_model, with op num: 19
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/test/src/./compiled_model/U280/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/test/src/./compiled_model/U280/customcnn.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is a42345b845b4961cdb5806f769ac4ed2, and has been saved to "/workspace/te
st/src/./compiled_model/U280/md5sum.txt"
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
+ echo -----------------------------------------
-----------------------------------------
+ echo 'COMPILE U280 COMPLETED'
COMPILE U280 COMPLETED
+ echo -----------------------------------------
-----------------------------------------


```

### Make target directory

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> bash -x make_target_u280.sh 
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MAKE TARGET U280 STARTED..'
MAKE TARGET U280 STARTED..
+ echo -----------------------------------------
-----------------------------------------
+ TARGET_U280=./target_u280
+ COMPILE_U280=./compiled_model/U280
+ APP=./application
+ NET_NAME=customcnn
+ rm -rf ./target_u280
+ mkdir -p ./target_u280/model_dir
+ cp ./application/app_mt.py ./target_u280
+ echo '  Copied application to TARGET_U280 folder'
  Copied application to TARGET_U280 folder
+ cp ./compiled_model/U280/customcnn.xmodel ./target_u280/model_dir/.
+ echo '  Copied xmodel file(s) to TARGET_U280 folder'
  Copied xmodel file(s) to TARGET_U280 folder
+ mkdir -p ./target_u280/images
+ python generate_images.py --dataset=mnist --image_dir=./target_u280/images --image_format=jpg --max_images=100000
2023-03-20 03:05:58.650459: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic li
brary 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_
LIBRARY_PATH: /opt/xilinx/xrt/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:/opt/vitis_ai/conda/envs/vitis-ai
-tensorflow/lib
2023-03-20 03:05:58.650509: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you d
o not have a GPU set up on your machine.
Command line options:
 --dataset      :  mnist
 --subset       :  test
 --image_dir    :  ./target_u280/images
 --image_list   :  
 --label_list   :  
 --image_format :  jpg
 --max_images   :  100000
+ echo '  Copied images to TARGET_U280 folder'
  Copied images to TARGET_U280 folder
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MAKE TARGET U280 COMPLETED'
MAKE TARGET U280 COMPLETED
+ echo -------------------------------------u280
-------------------------------------u280

```

## Run on Alveo U280 

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> source setup.sh DPUCAHX8H
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> cd target_u280
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> /usr/bin/python3 app_mt.py

Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  model_dir/customcnn.xmodel
Pre-processing 10000 images...
Starting 1 threads...
Throughput=3969.52 fps, total frames = 10000, time=2.5192 seconds
Correct:9807, Wrong:193, Accuracy:0.9807

```
