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

## GNN :

Then we implement a new model class Graph Neural Network which inherits from torch.nn.Module but we specify the forward function of the GNN. It should take as input a minibatch of adjacency matrices `A` (given as a 3-dimensional tensor of dimensions (minibatch_size , number_nodes , number_nodes) ) and return a matrix of size (minibatch_size,3) representing the scores for each example and predicted class.


The graph neural network is now tested on a simple graph classification task where the three classes correspond to ***star-shaped***, ***chain-shaped*** and ***random-shaped*** graphs. Because the GNN is more difficult to optimize and the dataset is smaller, we train the network for 500 epochs. We compare the GNN with a simple fully-connected network built directly on the adjacency matrix.

```
--------------------------------------------------
name:        DNN  train: 1.000  test: 0.829
--------------------------------------------------
name:        GNN  train: 1.000  test: 0.962
--------------------------------------------------

```

We observe that both networks are able to perfectly classify the training data, however, due to its particular structure, the graph neural network generalizes better to new data points.






