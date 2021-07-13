from matplotlib import pyplot as plt

import torchvision
from torchvision import datasets


def getdata():

    datasets.MNIST.resources = [('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'), ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
                                ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'), ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')]

    trainset = torchvision.datasets.MNIST(
        root='./data/', train=True, download=True)
    testset = torchvision.datasets.MNIST(
        root='./data/', train=False, download=True)

    Tr = trainset.targets.data.numpy()
    Tt = testset.targets.data.numpy()

    Xr = (trainset.data.float().view(-1, 784)/255.0).data.numpy()
    Xt = (testset.data.float().view(-1, 784)/255.0).data.numpy()

    return (Xr[Tr == 0][:100]*1.0, Xt[Tt == 0]*1.0, Xt[Tt != 0]*1.0)


def vis10(x):
    x = x.reshape(1, 10, 28, 28).transpose(0, 2, 1, 3).reshape(28, 280)
    plt.figure(figsize=(8, 1))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(x, cmap='gray')
    plt.show()
