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


782/782 [==============================] - 5s 6ms/step - loss: 0.3861 - accuracy: 0.8958 - val_loss: 0.1821 - val_accuracy: 0.9507
Epoch 2/10


782/782 [==============================] - 4s 5ms/step - loss: 0.1665 - accuracy: 0.9532 - val_loss: 0.1265 - val_accuracy: 0.9659
Epoch 3/10

782/782 [==============================] - 4s 5ms/step - loss: 0.1217 - accuracy: 0.9651 - val_loss: 0.1020 - val_accuracy: 0.9731
Epoch 4/10

782/782 [==============================] - 5s 6ms/step - loss: 0.0982 - accuracy: 0.9718 - val_loss: 0.0893 - val_accuracy: 0.9758
Epoch 5/10

782/782 [==============================] - 4s 5ms/step - loss: 0.0834 - accuracy: 0.9758 - val_loss: 0.0819 - val_accuracy: 0.9765
Epoch 6/10

782/782 [==============================] - 4s 6ms/step - loss: 0.0734 - accuracy: 0.9789 - val_loss: 0.0772 - val_accuracy: 0.9775
Epoch 7/10

782/782 [==============================] - 4s 6ms/step - loss: 0.0660 - accuracy: 0.9807 - val_loss: 0.0743 - val_accuracy: 0.9782
Epoch 8/10

782/782 [==============================] - 5s 6ms/step - loss: 0.0603 - accuracy: 0.9823 - val_loss: 0.0725 - val_accuracy: 0.9785
Epoch 9/10

782/782 [==============================] - 4s 5ms/step - loss: 0.0555 - accuracy: 0.9837 - val_loss: 0.0712 - val_accuracy: 0.9789
Epoch 10/10

782/782 [==============================] - 4s 6ms/step - loss: 0.0515 - accuracy: 0.9851 - val_loss: 0.0704 - val_accuracy: 0.9795

Save trained model to./models/float_model.h5.

Evaluate model on test dataset..

  1/157 [..............................] - ETA: 0s - loss: 0.0735 - accuracy: 0.9688
 22/157 [===>..........................] - ETA: 0s - loss: 0.0806 - accuracy: 0.9716
 44/157 [=======>......................] - ETA: 0s - loss: 0.0869 - accuracy: 0.9712
 67/157 [===========>..................] - ETA: 0s - loss: 0.0873 - accuracy: 0.9706
 89/157 [================>.............] - ETA: 0s - loss: 0.0764 - accuracy: 0.9742
111/157 [====================>.........] - ETA: 0s - loss: 0.0714 - accuracy: 0.9762
132/157 [========================>.....] - ETA: 0s - loss: 0.0642 - accuracy: 0.9788
153/157 [============================>.] - ETA: 0s - loss: 0.0613 - accuracy: 0.9803
157/157 [==============================] - 0s 2ms/step - loss: 0.0617 - accuracy: 0.9801
loss: 0.062
acc: 0.980
```

### Quantize the floating-point model

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src > python quantize.py 
Load float model..

Load Mnist dataset..
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

    8192/11490434 [..............................] - ETA: 0s
 5013504/11490434 [============>.................] - ETA: 0s
10870784/11490434 [===========================>..] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step

Run quantization..
[VAI INFO] Start CrossLayerEqualization...

 1/10 [==>...........................] - ETA: 0s
 3/10 [========>.....................] - ETA: 0s
 5/10 [==============>...............] - ETA: 0s
 7/10 [====================>.........] - ETA: 0s
 9/10 [==========================>...] - ETA: 0s
10/10 [==============================] - 0s 26ms/step
[VAI INFO] CrossLayerEqualization Done.
[VAI INFO] Start Quantize Calibration...


157/157 [==============================] - 2s 13ms/step
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

Load Mnist dataset..
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

    8192/11490434 [..............................] - ETA: 0s
 5013504/11490434 [============>.................] - ETA: 0s
10870784/11490434 [===========================>..] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step

Run quantization..
[VAI INFO] Start CrossLayerEqualization...

 1/10 [==>...........................] - ETA: 0s
 3/10 [========>.....................] - ETA: 0s
 5/10 [==============>...............] - ETA: 0s
 7/10 [====================>.........] - ETA: 0s
 9/10 [==========================>...] - ETA: 0s
10/10 [==============================] - 0s 26ms/step
[VAI INFO] CrossLayerEqualization Done.
[VAI INFO] Start Quantize Calibration...


157/157 [==============================] - 2s 13ms/step
[VAI INFO] Quantize Calibration Done.
[VAI INFO] Start Post-Quantize Adjustment...
[VAI INFO] Post-Quantize Adjustment Done.
[VAI INFO] Quantization Finished.

Saved quantized model as ./models/quantized_model.h5
```
### Finetuning
Here we just run finetuning once for demonstration. For further compiling we just used quantized_model.h5 generated before.

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > python finetune.py 

Load float model..

Load Mnist dataset..

Create quantize training model..
[INFO] Start CrossLayerEqualization...
10/10 [==============================] - 0s 33ms/step
[INFO] CrossLayerEqualization Done.

Compiling model..

Fit on Dataset..
Epoch 1/10
782/782 [==============================] - 48s 61ms/step - loss: 0.0077 - accuracy: 0.9978 - val_loss: 0.0738 - val_accuracy: 0.9882
Epoch 2/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0062 - accuracy: 0.9980 - val_loss: 0.0845 - val_accuracy: 0.9888
Epoch 3/10
782/782 [==============================] - 40s 51ms/step - loss: 0.0058 - accuracy: 0.9983 - val_loss: 0.0810 - val_accuracy: 0.9885
Epoch 4/10
782/782 [==============================] - 40s 51ms/step - loss: 0.0061 - accuracy: 0.9982 - val_loss: 0.0744 - val_accuracy: 0.9902
Epoch 5/10
782/782 [==============================] - 40s 51ms/step - loss: 0.0048 - accuracy: 0.9984 - val_loss: 0.0834 - val_accuracy: 0.9911
Epoch 6/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0047 - accuracy: 0.9986 - val_loss: 0.0807 - val_accuracy: 0.9893
Epoch 7/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0039 - accuracy: 0.9987 - val_loss: 0.0894 - val_accuracy: 0.9903
Epoch 8/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0034 - accuracy: 0.9989 - val_loss: 0.0863 - val_accuracy: 0.9904
Epoch 9/10
782/782 [==============================] - 39s 49ms/step - loss: 0.0042 - accuracy: 0.9989 - val_loss: 0.1043 - val_accuracy: 0.9893
Epoch 10/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0044 - accuracy: 0.9986 - val_loss: 0.0994 - val_accuracy: 0.9908

Saved finetuned model as ./models/finetuned_model.h5

Evaluate model on test Dataset..
157/157 [==============================] - 1s 7ms/step - loss: 0.0675 - accuracy: 0.9920
loss: 0.068
acc: 0.992

```

### Compile into DPU model file

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > bash -x compile_zcu102.sh 
+ ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
+ OUTDIR=./compiled_model
+ NET_NAME=customcnn
+ MODEL=./models/finetuned_model.h5
+ echo -----------------------------------------
-----------------------------------------
+ echo 'COMPILING MODEL FOR ZCU102..'
COMPILING MODEL FOR ZCU102..
+ echo -----------------------------------------
-----------------------------------------
+ compile
+ tee compile.log
+ vai_c_tensorflow2 --model ./models/finetuned_model.h5 --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json --output_dir ./compiled_model --net_name customcnn
/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.7/site-packages/xnnc/translator/tensorflow_translator.py:1843: H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
  value = param.get(group).get(ds).value
[INFO] parse raw model     :100%|██████████| 10/10 [00:00<00:00, 16871.70it/s]               
[INFO] infer shape (NHWC)  :100%|██████████| 26/26 [00:00<00:00, 2956.30it/s]                
[INFO] generate xmodel     :100%|██████████| 26/26 [00:00<00:00, 5561.60it/s]                
[INFO] Namespace(inputs_shape=None, layout='NHWC', model_files=['./models/finetuned_model.h5'], model_type='tensorflow2', out_filename='./compiled_model/customcnn_org.xmodel', proto=None)
[INFO] tensorflow2 model: models/finetuned_model.h5
[OPT] No optimization method available for xir-level optimization.
[INFO] generate xmodel: /workspace/myproj/tf2-mnist-end-to-end/compiled_model/customcnn_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/vitis-ai-user/log/xcompiler-20210325-093926-3120"
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Graph name: mnist_customcnn_model, with op num: 42
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/myproj/tf2-mnist-end-to-end/./compiled_model/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/myproj/tf2-mnist-end-to-end/./compiled_model/customcnn.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 764142e83d074ea9470b9eb9d0757f68, and been saved to "/workspace/myproj/tf2-mnist-end-to-end/./compiled_model/md5sum.txt"
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MODEL COMPILED'
MODEL COMPILED
+ echo -----------------------------------------
-----------------------------------------

```

### Make target directory

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > bash -x make_target_zcu102.sh 
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MAKE TARGET ZCU102 STARTED..'
MAKE TARGET ZCU102 STARTED..
+ echo -----------------------------------------
-----------------------------------------
+ TARGET_ZCU102=./target_zcu102
+ COMPILE_ZCU102=./compiled_model
+ APP=./application
+ NET_NAME=customcnn
+ rm -rf ./target_zcu102
+ mkdir -p ./target_zcu102/model_dir
+ cp ./application/app_mt.py ./target_zcu102
+ echo '  Copied application to TARGET_ZCU102 folder'
  Copied application to TARGET_ZCU102 folder
+ cp ./compiled_model/customcnn.xmodel ./target_zcu102/model_dir/.
+ echo '  Copied xmodel file(s) to TARGET_ZCU102 folder'
  Copied xmodel file(s) to TARGET_ZCU102 folder
+ mkdir -p ./target_zcu102/images
+ python generate_images.py --dataset=mnist --image_dir=./target_zcu102/images --image_format=jpg --max_images=10000
2021-03-25 09:42:34.445257: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Command line options:
 --dataset      :  mnist
 --subset       :  test
 --image_dir    :  ./target_zcu102/images
 --image_list   :  
 --label_list   :  
 --image_format :  jpg
 --max_images   :  10000
+ echo '  Copied images to TARGET_ZCU102 folder'
  Copied images to TARGET_ZCU102 folder
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MAKE TARGET ZCU102 COMPLETED'
MAKE TARGET ZCU102 COMPLETED
+ echo -----------------------------------------
-----------------------------------------

```

## Run on Alveo U280 
source setup.sh DPUCAHX8H
cd build/target_u280
/usr/bin/python3 app_mt.py

```

```
