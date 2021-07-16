## CCA Primal Formulation :

the CCA problem in its primal form consists of maximizing the cross-correlation objective:

![image](https://user-images.githubusercontent.com/85687148/125965316-58648548-29a6-40c9-b0a0-09abfb18b847.png)

subject to autcorrelation constraints :
![image](https://user-images.githubusercontent.com/85687148/125965441-a6b2d142-df43-4e17-9a8c-4cdc427825c8.png)
and:
![image](https://user-images.githubusercontent.com/85687148/125965533-831fc418-51bc-4156-9195-93ac46b21e84.png)


Using the method of Lagrange multipliers, this optimization problem can be reduced to finding the first eigenvector of the generalized eigenvalue problem:


![image](https://user-images.githubusercontent.com/85687148/125965899-457cd641-d724-4ad8-a887-a36f7de97ab2.png)

