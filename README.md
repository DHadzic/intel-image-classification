# Intel image classification

### Author: Damir Hadzic  

### Subject: Neural network

## Goal

Train set contains 14000 images.  

Test set contains 2500 images.  

Total number of classes is 6 (mountain, street, glacier, buildings, sea, forest).  

Train a neural network model with high success rate in class prediction of given image.  

## Packages

Package versions:  
* tensorflow - __*2.8.2*__  
* opencv - __*4.6.0.66*__  
* matplotlib - __*3.5.3*__

## Models

* Simple CNN 1 (`models/simple-14k-71p`)

![Architecture](./assets/simple1.png)

Model trained on basic dataset. Architecture is shown above. Number of epoches is 20. Accuracy of given model is **71.6%**

* Simple CNN 2 (`models/simple-18k-70p`)

Model trained on basic dataset + noise images created on 25% of the dataset. Same architecture is used as in previous model. Number of epoches is 30 for this dataset. Accuracy of given model is **70.8%**
