# DEPLOY OF A SIMPLE CNN ON ALVEO U280

Here is presented an example of FPGA neural network inference deployed on alveo u280 board with Vivado 2020.1. The Dataset used for classification is the MNIST dataset. The NN used is a simple CNN: 
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 convolution_layer_1 (Conv2D  (None, 26, 26, 12)       120       
 )                                                               
                                                                 
 max_pooling_layer_1 (MaxPoo  (None, 13, 13, 12)       0         
 ling2D)                                                         
                                                                 
 flatten_layer (Flatten)     (None, 2028)              0         
                                                                 
 dense_layer (Dense)         (None, 10)                20290     
                                                                 
 softmax (Activation)        (None, 10)                0         
                                                                 
=================================================================
Total params: 20,410
Trainable params: 20,410
Non-trainable params: 0
```

## Software Requirements
* Vivado and Vitis 2020.1 
* Platform files for your board (2020.1 or older)
* XRT (2020.1 or newer) 
* Python 3.7

## Run the Virtual Environment
All the libraries and the source of the SWs are in the virtual environment.
```
[carini@localhost test]$  ./script.sh

Epoch 1/15
422/422 [==============================] - 3s 7ms/step - loss: 0.4896 - accuracy: 0.8674 - val_loss: 0.2208 - val_accuracy: 0.9388
Epoch 2/15
422/422 [==============================] - 3s 7ms/step - loss: 0.2337 - accuracy: 0.9317 - val_loss: 0.1744 - val_accuracy: 0.9515
Epoch 3/15
422/422 [==============================] - 3s 7ms/step - loss: 0.1720 - accuracy: 0.9511 - val_loss: 0.1263 - val_accuracy: 0.9657
Epoch 4/15
422/422 [==============================] - 3s 7ms/step - loss: 0.1356 - accuracy: 0.9617 - val_loss: 0.1042 - val_accuracy: 0.9717
Epoch 5/15
422/422 [==============================] - 3s 7ms/step - loss: 0.1109 - accuracy: 0.9691 - val_loss: 0.0931 - val_accuracy: 0.9743
Epoch 6/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0951 - accuracy: 0.9734 - val_loss: 0.0861 - val_accuracy: 0.9773
Epoch 7/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0831 - accuracy: 0.9764 - val_loss: 0.0753 - val_accuracy: 0.9812
Epoch 8/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0741 - accuracy: 0.9793 - val_loss: 0.0721 - val_accuracy: 0.9800
Epoch 9/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0687 - accuracy: 0.9797 - val_loss: 0.0693 - val_accuracy: 0.9823
Epoch 10/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0626 - accuracy: 0.9818 - val_loss: 0.0625 - val_accuracy: 0.9838
Epoch 11/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0577 - accuracy: 0.9837 - val_loss: 0.0671 - val_accuracy: 0.9830
Epoch 12/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0545 - accuracy: 0.9841 - val_loss: 0.0635 - val_accuracy: 0.9840
Epoch 13/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0512 - accuracy: 0.9853 - val_loss: 0.0595 - val_accuracy: 0.9838
Epoch 14/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0483 - accuracy: 0.9859 - val_loss: 0.0598 - val_accuracy: 0.9843
Epoch 15/15
422/422 [==============================] - 3s 7ms/step - loss: 0.0460 - accuracy: 0.9860 - val_loss: 0.0598 - val_accuracy: 0.9840


Size of the Keras model: 0.233736 MB
313/313 [==============================] - 1s 2ms/step
Keras test Accuracy: 0.9805

........

```



## hls4ml Configuration
In this tool two fundamental parameters are the reuse factor and the precision used for the operations. 
```
config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = 1
    config['Model']['Strategy'] = 'Resource'
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    
    config['LayerName']['layer0']['Precision'] = 'ap_ufixed<8,3>'

    config['LayerName']['dense_layer']['ReuseFactor'] = 13
    config['LayerName']['softmax']['Strategy']    = 'Stable'

    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True


    cfg = hls4ml.converters.create_config(board='alveo-u280', clock_period= 5, part='xcu280-fsvh2892-2L-e', backend='VivadoAccelerator')
    cfg['HLSConfig'] = config

    cfg['AcceleratorConfig']['Driver']    = 'python'
    cfg['AcceleratorConfig']['Board']     = 'alveo-u280'
    cfg['AcceleratorConfig']['Interface'] = 'axi_stream'
    cfg['AcceleratorConfig']['Precision']['Input']  = 'float'
    cfg['AcceleratorConfig']['Precision']['Output'] = 'float'
    cfg['IOType']= 'io_stream'
    cfg['KerasModel'] = model
    cfg['OutputDir'] = PROJECT_PATH + 'my_simpel_CNN_test/hls4ml_prj'


```

## General results

The tests are performed on a Alveo u280 board.  
 
### Vivado HLS resource utilization [MNIST classification]

|Model                |BRAM[Blocks]|DSP|FF[k]|LUT[k]|
|---------------------|------------|---|-----|------|
|Platform             |/           |/  |/    |/     |
|CNN  |734         |1598 |121.1 |152.5 |
 
### Post-implementation resource utilization [MNIST classification] - from Vitis_analyzer tool

|Model                |BRAM[Blocks]|DSP|FF[k]|LUT[k]|
|---------------------|------------|---|-----|------|
|Platform             |202         |4  |148.5|102.0 |
|CNN  |22          |1598 |113.0 |118.5  |

### NN performance
  
|MODEL              |Accuracy [\%]|Rate[Images\s]|t<sub>img</sub>[&#956;s]|
|-------------------|-------------|--------------|------------------------|
|CNN                |97.66        |60519.5         |          16.5            |

  
### Comparisons
  
Here are presented the prediction times for different devices (10000 samples dataset).

|Device             |t<sub>img</sub><sup>CNN</sup>[&#956;s]|
|-------------------|--------------------------------------|
|CPU [AMD Ryzen 7 4800u]         |117                                                                   
|CPU [AMD EPYC 7282]       |74.2                             |
|ALVEO[u280]       |16.5                                     |
