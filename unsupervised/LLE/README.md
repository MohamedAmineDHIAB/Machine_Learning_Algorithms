## Implementing Locally Linear Embedding

we use the sklearn swiss roll dataset with `N=1000` data points and a noise parameter of `0.25`:

![image](https://user-images.githubusercontent.com/85687148/125962328-6abaa684-f514-46ce-87b0-d9d6b6ddcac1.png)


Although the dataset is in three dimensions, the points follow a two-dimensional low-dimensional structure. The goal of embedding algorithms is to extract this underlying structure, in this case, unrolling the swiss roll into a two-dimensional Euclidean space.

We try to find this two-dimensional space by applying LLE on the swiss roll dataset and vary the noise in the data and the parameter `k` of the LLE algorithm (which corresponds to the number of neighbors used in the algorithm). 

#### Results are shown below:

![image](https://user-images.githubusercontent.com/85687148/125962505-9afc7f11-df21-4dc5-b4b3-35142542631a.png)




