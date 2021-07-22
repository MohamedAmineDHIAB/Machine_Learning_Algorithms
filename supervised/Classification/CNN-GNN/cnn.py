import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import utils
import numpy

import matplotlib


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    Xr, Tr = trainset.data.float().view(-1, 1, 28, 28)/127.5-1, trainset.targets
    Xt, Tt = testset.data.float().view(-1, 1, 28, 28)/127.5-1, testset.targets
    return(Xr, Tr, Xt, Tt)

if __name__=='__main__':
    Xr, Tr, Xt, Tt=get_data()
