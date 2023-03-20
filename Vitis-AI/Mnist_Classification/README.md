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

--------------------------------------------------------------------------------
782/782 [==============================] - 5s 6ms/step - loss: 0.3861 - accuracy: 0.8958 - val_loss: 0.1821 - val_accuracy: 0.9507
Epoch 2/10

--------------------------------------------------------------------------------
782/782 [==============================] - 4s 5ms/step - loss: 0.1665 - accuracy: 0.9532 - val_loss: 0.1265 - val_accuracy: 0.9659
Epoch 3/10
--------------------------------------------------------------------------------
782/782 [==============================] - 4s 5ms/step - loss: 0.1217 - accuracy: 0.9651 - val_loss: 0.1020 - val_accuracy: 0.9731
Epoch 4/10
--------------------------------------------------------------------------------
782/782 [==============================] - 5s 6ms/step - loss: 0.0982 - accuracy: 0.9718 - val_loss: 0.0893 - val_accuracy: 0.9758
Epoch 5/10
--------------------------------------------------------------------------------
782/782 [==============================] - 4s 5ms/step - loss: 0.0834 - accuracy: 0.9758 - val_loss: 0.0819 - val_accuracy: 0.9765
Epoch 6/10
--------------------------------------------------------------------------------
782/782 [==============================] - 4s 6ms/step - loss: 0.0734 - accuracy: 0.9789 - val_loss: 0.0772 - val_accuracy: 0.9775
Epoch 7/10
--------------------------------------------------------------------------------
782/782 [==============================] - 4s 6ms/step - loss: 0.0660 - accuracy: 0.9807 - val_loss: 0.0743 - val_accuracy: 0.9782
Epoch 8/10
--------------------------------------------------------------------------------
782/782 [==============================] - 5s 6ms/step - loss: 0.0603 - accuracy: 0.9823 - val_loss: 0.0725 - val_accuracy: 0.9785
Epoch 9/10
--------------------------------------------------------------------------------
782/782 [==============================] - 4s 5ms/step - loss: 0.0555 - accuracy: 0.9837 - val_loss: 0.0712 - val_accuracy: 0.9789
Epoch 10/10
--------------------------------------------------------------------------------
782/782 [==============================] - 4s 6ms/step - loss: 0.0515 - accuracy: 0.9851 - val_loss: 0.0704 - val_accuracy: 0.9795

Save trained model to./models/float_model.h5.

Evaluate model on test dataset..

  1/157 [..............................] - ETA: 0s - loss: 0.0735 - accuracy: 0.9688--------------------------------------------------------------------------------
 22/157 [===>..........................] - ETA: 0s - loss: 0.0806 - accuracy: 0.9716---------------------------------------------------------------------------------------
 44/157 [=======>......................] - ETA: 0s - loss: 0.0869 - accuracy: 0.9712--------------------------------------------------------------------------------
 67/157 [===========>..................] - ETA: 0s - loss: 0.0873 - accuracy: 0.9706--------------------------------------------------------------------------------
 89/157 [================>.............] - ETA: 0s - loss: 0.0764 - accuracy: 0.9742--------------------------------------------------------------------------------
111/157 [====================>.........] - ETA: 0s - loss: 0.0714 - accuracy: 0.9762--------------------------------------------------------------------------------
132/157 [========================>.....] - ETA: 0s - loss: 0.0642 - accuracy: 0.9788--------------------------------------------------------------------------------
153/157 [============================>.] - ETA: 0s - loss: 0.0613 - accuracy: 0.9803--------------------------------------------------------------------------------
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

    8192/11490434 [..............................] - ETA: 0s-------------------------------------------------------------
 5013504/11490434 [============>.................] - ETA: 0s-------------------------------------------------------------
10870784/11490434 [===========================>..] - ETA: 0s-------------------------------------------------------------
11493376/11490434 [==============================] - 0s 0us/step

Run quantization..
[VAI INFO] Start CrossLayerEqualization...

 1/10 [==>...........................] - ETA: 0s-------------------------------------------------------------
 3/10 [========>.....................] - ETA: 0s-------------------------------------------------------------
 5/10 [==============>...............] - ETA: 0s-------------------------------------------------------------
 7/10 [====================>.........] - ETA: 0s------------------------------------------------------------
 9/10 [==========================>...] - ETA: 0s-------------------------------------------------------------
10/10 [==============================] - 0s 26ms/step
[VAI INFO] CrossLayerEqualization Done.
[VAI INFO] Start Quantize Calibration...

-------------------------------------------------------------
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

Load quantized model..

Load Mnist dataset..

Compile model..

Evaluate model on test Dataset
---------------------------------------------------------------------------
157/157 [==============================] - 1s 4ms/step - loss: 0.0614 - accuracy: 0.9800
loss: 0.061
acc: 0.980
```
### Finetuning
Here we just run finetuning once for demonstration. For further compiling we just used quantized_model.h5 generated before.

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src > python finetune.py 



```

### Compile into DPU model file

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src > bash -x compile_u280.sh 

```

### Make target directory

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> bash -x make_target_u280.sh 

```

## Run on Alveo U280 

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> source setup.sh DPUCAHX8H
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> cd target_u280
(vitis-ai-tensorflow2) Vitis-AI /workspace/test/src> /usr/bin/python3 app_mt.py


```
