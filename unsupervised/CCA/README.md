## CCA Primal Formulation :

the CCA problem in its primal form consists of maximizing the cross-correlation objective:

![image](https://user-images.githubusercontent.com/85687148/125965316-58648548-29a6-40c9-b0a0-09abfb18b847.png)

subject to autcorrelation constraints :


![image](https://user-images.githubusercontent.com/85687148/125965441-a6b2d142-df43-4e17-9a8c-4cdc427825c8.png)


and:


![image](https://user-images.githubusercontent.com/85687148/125965533-831fc418-51bc-4156-9195-93ac46b21e84.png)


Using the method of Lagrange multipliers, this optimization problem can be reduced to finding the first eigenvector of the generalized eigenvalue problem:


![image](https://user-images.githubusercontent.com/85687148/125965899-457cd641-d724-4ad8-a887-a36f7de97ab2.png)

## CCA Primal Data :

![image](https://user-images.githubusercontent.com/85687148/125966128-0f597278-7b9d-4279-8ba3-ae1cb82e0fa9.png)


## CCA Primal Eigenvectors :

![image](https://user-images.githubusercontent.com/85687148/125966144-1222292c-7052-47ae-a538-a1b818e3d281.png)


## CCA Primal Projections :

![image](https://user-images.githubusercontent.com/85687148/125966158-3635e6f8-a8e8-44dd-9cb6-a2645ba73e92.png)


## CCA Dual Data :

Dual Formulation can be used in case of high dimensional input (d>N) where N is the number of samples and d is the dimension of one sample.
We consider the scenario where sources emit spatially, and two (noisy) receivers measure the spatial field at different locations. We would like to identify signal that is common to the two measured locations, e.g. a given source emitting at a given frequency. 

#### We first load the data and show the first example:

![image](https://user-images.githubusercontent.com/85687148/125966218-ca8ab34b-2cbb-41f7-a1bf-702326b15d71.png)


## CCA Dual Eigenvectors :

![image](https://user-images.githubusercontent.com/85687148/125966237-609d6e6a-efa6-4bbf-8583-8a22f88e335b.png)



## CCA Dual Projections :

![image](https://user-images.githubusercontent.com/85687148/125966248-72f929b2-b73b-4ce1-9121-00a7a47ed536.png)
