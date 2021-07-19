## Gaussian Processes

In this repository, we implement Gaussian process regression and apply it to a toy and a real dataset (Yacht Hydrodynamics).


### Toy Dataset:



#### Effects of width and noise :

Here we compute mean and variance of the prediction at every location of the input space for the toy dataset and compares the behavior of the Gaussian process for various noise parameters and width parameters :

![image](https://user-images.githubusercontent.com/85687148/126000259-c08e1a13-e69f-4a6c-b963-4bc81257623b.png)


### Yacht Hydrodynamics:

#### Effects of width and noise :

Here we compute the loglikelihood of the train and the test sets for the yacht hydrodynamics dataset as a function of various noise parameters and width parameters , in the end we plot the contours as follow :

![image](https://user-images.githubusercontent.com/85687148/126226118-ddc0fe2d-913b-4463-aa3c-a0e17a357984.png)
