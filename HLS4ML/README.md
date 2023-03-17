# ALVEO_ML

Here is presented an example of FPGA neural network inference deployed on alveo u280 board with Vivado 2020.1. The NN used is a simple CNN: 

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
|CNN  |365         |10 |68.8 |129.0 |
 
### Post-implementation resource utilization [MNIST classification]

|Model                |BRAM[Blocks]|DSP|FF[k]|LUT[k]|
|---------------------|------------|---|-----|------|
|Platform             |178         |4  |123.4|100.2 |
|CNN  |42          |10 |28.7 |22.6  |

### NN performance
  
|MODEL              |Accuracy [\%]|Rate[Images\s]|t<sub>img</sub>[&#956;s]|
|-------------------|-------------|--------------|------------------------|
|CNN                |97.11        |52600         |19                      |

  
### Comparisons
  
Here are presented the prediction times for different devices (10000 samples dataset).

|Device             |t<sub>img</sub><sup>CNN</sup>[&#956;s]|t<sub>img</sub><sup>DNN</sup>[&#956;s]|
|-------------------|--------------------------------------|--------------------------------------|
|CPU [AMD Ryzen 7 4800u]         |95                                    |24                                    |
|CPU [AMD EPYC 7282]       |30                                    |22                                    |
|ALVEO[u280]       |87                                    |85                                    |
