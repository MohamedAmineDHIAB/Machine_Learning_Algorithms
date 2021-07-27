# Principal Component Analysis

PCA is a dimensionality reduction technique.

In this REPO we use the following IRIS dataset with ***4*** features and a corresponding `target` to each feature vector :

![image](https://user-images.githubusercontent.com/85687148/127187060-9258466c-3829-4289-ac8d-089f2b4f5182.png)

Then we implement PCA using NumPy and apply it to the IRIS dataset getting in the end a dataset with feature vectors dimension `d=2` which corresponds to the projection of our original dataset along the Eigenvectors with the highest 2 Eigenvalues from the SVD decomposition of the covariance matrix . We get the following plot :

![image](https://user-images.githubusercontent.com/85687148/127187771-a2460a9f-a3a8-428f-b8f5-0815bce252bf.png)

