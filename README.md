# AI_Accelerators_MSc_Thesis
This is a repository for my master thesis on AI hardware accelerators developed @ Politecnico di Milano.

 * **Advisor**: Prof. Christian Pilato
 * **Co-Advisor**: Ing. Mattia Tibaldi
 
 
 
<!-- DESCRIPTION -->
## Description
In this repository there is the material used in the master thesis and stored on the server in DEIB. For the synthesis and implementation of CNNs on hardware are used two frameworks: HLS4ML and vitis-AI. In the thesis there is a focus on the comparison between the two tools trying to figure out which one is better to use in different possible scenarios. 
With hls4ml, the xclbin is generated each time starting from the model that is given to it as input. Usually the generation of this bitstream takes about 3 hours. While in Vitis-AI an already existing bitstream is used and the model is compiled as if it were micro-code. 
For more details on how the tools work, go to the relative folders (<a href="https://github.com/davidecarini/AI_Accelerators_MSc_Thesis/tree/main/HLS4ML">HLS4ML</a>, <a href="https://github.com/davidecarini/AI_Accelerators_MSc_Thesis/tree/main/Vitis-AI">Vitis-AI</a>) .

The dataset used for training and evaluation of the neural network is the MNIST (https://en.wikipedia.org/wiki/MNIST_database). 



<!-- MODEL --> 
## Neural Network
The neural network used in the thesis is a Convolutional Neural Network and it has the following structure: 
```
Model: "mnist_cnn_model"
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

Training time: 43,2 seconds
Size of the Keras model: 0.233736 MB
Keras test Accuracy: 98,05 %
```


<!-- COMPARISON -->
## Comparison between CNN on HLS4ML and Vitis-AI [Alveo U280]

|TOOL              |Total images [Images] |Test Accuracy [\%]|Throughput [Images\s]|t <sub>img</sub>[&#956;s]|
|------------------|----------------------|-------------|--------------|------------------------|
|HLS4ML          | 10000     |97.66        |60519.5         |          251.92            |
|Vitis-AI        | 10000      |  98.07     |3969.52         |          16.5            |




<!-- LINKS -->
## Links
<table style="margin-left: auto; margin-right: auto">
<thead>
<tr><th>DESCRIPTION</th><th>LINK</th></tr>
</thead>
<tbody>
<tr><td align="center">Meeting Notes</td><td> https://www.notion.so/030e0768ad6d4bb3bb5c99557ac8c06a?v=c46ee3523e3c481a92231fe94cc00bd3 </td></tr>
<tr><td align="center">LaTeX Thesis</td><td>https://www.overleaf.com/project/638883ef1f6f113398139581</td></tr>
<tr><td align="center">Colab</td><td>https://colab.research.google.com/drive/1pKZtZ9_iotdf0YHwzCta6M2bcLMQDrCG</td></tr>
<tr><td align="center">Final Presentation</td><td></td></tr>
</tbody>
</table>




<!-- AUTHORS -->
## Author
<table style="margin-left: auto; margin-right: auto">
<thead>
<tr><th>NAME</th><th>EMAIL</th><th>PERSON CODE</th><th>STUDENT ID</th></tr>
</thead>
<tbody>
<tr><td><a href="https://github.com/davidecarini">Davide Carini<a/></td><td align="center">davide.carini@mail.polimi.it</td><td>10568649</td><td>976571</td></tr>
</tbody>
</table>



## Acknowledgments
For the development of this thesis I relied on these very useful repositories: 
* https://github.com/selwyn96/Alveo-tutorial/blob/main/training/MNIST_Test/MNIST_train.ipynb  
* https://github.com/selwyn96/hls4ml
* https://github.com/lobster1989/Mnist-classification-Vitis-AI-1.3-TensorFlow2



## Contribute
Contributions and discovery of any bug are always welcome!

