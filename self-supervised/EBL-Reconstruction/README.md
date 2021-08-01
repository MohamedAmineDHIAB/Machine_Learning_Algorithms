# MNIST Inpainting with Energy-Based Learning


In this REPO, we consider the task of inpainting of incomplete handwritten digits, and for this, we would like to make use of neural networks and the ***Energy-Based Learning*** framework.

We first start by applying masks of certain dimension and on random positions to MNIST data samples like the following:

![image](https://user-images.githubusercontent.com/85687148/126914476-6c8091e3-0090-4643-8152-395382a7b3da.png)


## Baseline : PCA Reconstruction

A simple technique for impainting an image is principal component analysis. It consists of taking the incomplete image and projecting it on the `d` principal components of the training data.

The PCA-based inpainting technique is tested below on 10 test points for which a patch is missing. We observe that the patch-like perturbation is less severe when `d` is low, but the reconstructed part of the digit appears blurry. Conversely, if setting `d` high, more details become available, but the missing pattern appears more prominent.

We get the following results for :


- `d=10`
![image](https://user-images.githubusercontent.com/85687148/127785415-a2886545-f929-4946-82cd-3beddd7ee5a8.png)


- `d=60`
![image](https://user-images.githubusercontent.com/85687148/127785419-5989f0b7-09f8-4a7e-b6b1-d6fcc2172d3d.png)


- `d=360`
![image](https://user-images.githubusercontent.com/85687148/127785422-d1d9c76b-bd04-4544-b7bb-73e41b74a549.png)




