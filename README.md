# AI_Accelerators_MSc_Thesis
This is a repository for my master thesis on AI hardware accelerators developed @ Politecnico di Milano.

 * **Advisor**: Prof. Christian Pilato
 * **Co-Advisor**: Ing. Mattia Tibaldi
 
 
 
<!-- DESCRIPTION -->
## Description
In this repository there is the material used in the master thesis and stored on the server in DEIB. For the synthesis and implementation of CNNs on hardware are used two frameworks: HLS4ML and vitis AI. In the thesis there is a focus on the comparison between the two tools trying to figure out which one is better to use in different possible scenarios. 
With hls4ml, the xclbin is generated each time starting from the model that is given to it as input. Usually the generation of this bitstream takes about 3 hours. While in Vitis-AI an already existing bitstream is used and the model is compiled as if it were micro-code. The dataset used for training and evaluation of the neural network is the MNIST (https://en.wikipedia.org/wiki/MNIST_database).



<!-- MODEL --> 
## Neural Network
The neural network used in the thesis has the following structure: 

<img src="https://github.com/davidecarini/AI_Accelerators_MSc_Thesis/blob/main/images/Keras_model.png" alt="CNN model" style="float: left; margin-right: 100px;" />



<!-- COMPARISON -->
## Comparison between CNN on HLS4ML and Vitis-AI

|TOOL              |Total images [Images] |Accuracy [\%]|Throughput[Images\s]|t<sub>img</sub>[&#956;s]|
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
## Authors
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
Contributions are always welcome!

