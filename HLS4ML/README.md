# ALVEO_ML

Here is presented an example of FPGA neural network inference deployed on alveo u280 board with Vivado 2020.1. The NN used is a simple CNN: 

![CNN](https://github.com/davidecarini/AI_Accelerators_MSc_Thesis/blob/main/images/Keras_model.png)

The Dataset used for classification is the MNIST dataset.
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
 
### Post-implementation resource utilization [MNIST classification]

|Model                |BRAM[Blocks]|DSP|FF[k]|LUT[k]|
|---------------------|------------|---|-----|------|
|Platform             |178         |4  |123.4|100.2 |
|CNN  |42          |10 |28.7 |22.6  |

### NN performance
  
|MODEL              |Accuracy [\%]|Rate[Images\s]|t<sub>img</sub>[&#956;s]|
|-------------------|-------------|--------------|------------------------|
|CNN                |97.66        |60519.5         |          16.5            |

  
### Comparisons
  
Here are presented the prediction times for different devices (10000 samples dataset).

|Device             |t<sub>img</sub><sup>CNN</sup>[&#956;s]|
|-------------------|--------------------------------------|
|CPU [AMD Ryzen 7 4800u]         |129                                                                   
|CPU [AMD EPYC 7282]       |74.2                             |
|ALVEO[u280]       |16.5                                     |
