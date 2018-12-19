# 3D-CNN for 2D image classification
This is an implementation of 2D image classification with 3D-CNN(Basic-block ResNet).

91% accuracy achieved on CIFAR-10 with 0.36M(360,065) parameters in 25 epochs of training.

## Concept
In regular CNN, feature maps get shrinked with pooling as the process goes into deeper layers.
With shrinked feature maps, you need bigger channel sizes to keep enough information in them.
This is causing serious problem because CNN parameters count and computational cost increase in order of channels^2. 

In this method, we handle the feature maps as 3D data and expand it along the third axis instead of increasing channels.
Each convolution on these 3D featrure maps is 3D-convolution.

## Run
Tested on Python3 + Chainer(v5). 

        python3 train.py -g 0 -b 256 -e 25 -l 0.005 -w 0.0005

## TODO
- Implement SqueezeNet as 3D CNN.
- Parameter count comparison over 2D-CNN.