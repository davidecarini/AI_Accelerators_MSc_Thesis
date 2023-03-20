# DEPLOY OF CNN ON ALVEO U280

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
Vivado and Vitis 2020.1 <br />
Platform files for your board (2020.1 or older) <br />
XRT (2020.1 or newer) <br />
Python 3.7

## Run the Virtual Environment
All the libraries and the source of the SWs are in the virtual environment.
```
./script.sh
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
