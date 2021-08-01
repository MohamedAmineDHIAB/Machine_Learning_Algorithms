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

## Energy-Based Learning :

We now consider the energy-based learning framework where we learn an energy function to discriminate between correct and incorrect reconstructions. However to be able to generate images which are very close to the correct reconstructions we need to have a model that generates reconstructed images that are still plausible enough to confuse the energy-based model and for which meaningful gradient signal can be extracted, for that we consider a generator network that takes as input the incomplete images.

These are the architectures of the ***Generative*** model and the ***Energy-Based*** model :

`
    enet = nn.Sequential(
        nn.Linear(784, 256), nn.Hardtanh(),
        nn.Linear(256, 256), nn.Hardtanh(),
        nn.Linear(256, 1),
    )
`
`
    gnet = nn.Sequential(
        nn.Linear(784, 256), nn.Hardtanh(),
        nn.Linear(256, 256), nn.Hardtanh(),
        nn.Linear(256, 784), nn.Hardtanh()
    )
    train(enet, gnet, epochs=50)
`

![image](https://user-images.githubusercontent.com/85687148/127786427-c9b12b40-bf44-4838-b773-a21913813bf9.png)



These are the results that we get,

***- Before applying the model :***

![image](https://user-images.githubusercontent.com/85687148/127786343-1abf5abe-3604-47fa-a10b-b20aa826262b.png)

***- After applying the model :***

![image](https://user-images.githubusercontent.com/85687148/127786341-b31c68ef-50f4-4fd8-b30c-e67c7e5d728e.png)





