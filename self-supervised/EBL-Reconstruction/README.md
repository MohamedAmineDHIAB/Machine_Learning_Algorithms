# MNIST Inpainting with Energy-Based Learning


In this REPO, we consider the task of inpainting of incomplete handwritten digits, and for this, we would like to make use of neural networks and the ***Energy-Based Learning*** framework.

![image](https://user-images.githubusercontent.com/85687148/126914476-6c8091e3-0090-4643-8152-395382a7b3da.png)


## Baseline : PCA Reconstruction 

A simple technique for impainting an image is principal component analysis. It consists of taking the incomplete image and projecting it on the `d` principal components of the training data.

The PCA-based inpainting technique is tested below on 10 test points for which a patch is missing. We observe that the patch-like perturbation is less severe when `d` is low, but the reconstructed part of the digit appears blurry. Conversely, if setting `d` high, more details become available, but the missing pattern appears more prominent.



