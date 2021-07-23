# Convolutional Neural Networks & Graph Neural Networks

In this REPO, we train a collection of neural networks including a convolutional neural network on the MNIST dataset, and a graph neural network on some graph classification task.

## CNN :

We first consider the convolutional neural network, which we apply to the MNIST data.
We consider for this dataset a convolution network made of four convolutions and two pooling layers.
The network is wrapped in the class `utils.NNClassifier`, which exposes scikit-learn-like functions such as `fit()` and `predict()`. To evaluate the ***convolutional neural network***, we also consider two simpler baselines: a ***one-layer linear network***, and ***standard fully-connected network*** composed of two layers.


We train each classifier for `5 epochs` and print the classification accuracy on the training and test data (i.e. the fraction of the examples that are correctly classified). 
We observe that the convolutional neural network reaches the higest accuracy with less than ***2%*** of misclassified digits on the test data.

```
----------------------------------------------------------------------
    linear accuracy on train: 0.908  accuracy on test: 0.907
----------------------------------------------------------------------
      full accuracy on train: 0.973  accuracy on test: 0.969
----------------------------------------------------------------------
      conv accuracy on train: 0.990  accuracy on test: 0.987

```



These are the digits that were predicted with the ***highest*** probability by the ***cnn*** :

![image](https://user-images.githubusercontent.com/85687148/126724962-dba4324b-49a7-4505-8e2e-cdb8c0ff71a7.png)

These are the digits that were predicted with the ***lowest*** probability by the ***cnn*** :

![image](https://user-images.githubusercontent.com/85687148/126724997-52f695bd-a489-40c6-85a9-bee35b770ba4.png)





