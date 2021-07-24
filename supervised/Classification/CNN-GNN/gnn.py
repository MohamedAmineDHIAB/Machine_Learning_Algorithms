import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import utils
import numpy
import cnn
from matplotlib import pyplot as plt


class GNN(torch.nn.Module):

    def __init__(self, nbnodes, nbhid, nbclasses):
        torch.nn.Module.__init__(self)
        self.m = nbnodes
        self.h = nbhid
        self.c = nbclasses

        self.U = torch.nn.Parameter(torch.FloatTensor(
            numpy.random.normal(0, nbnodes**-.5, [nbnodes, nbhid])))
        self.W = torch.nn.Parameter(torch.FloatTensor(
            numpy.random.normal(0, nbhid**-.5, [nbhid, nbhid])))
        self.V = torch.nn.Parameter(torch.zeros([nbhid, nbclasses]))

    def forward(self, A):

        return (Y)


if __name__ == '__main__':
    Xr, Tr, Xt, Tt = cnn.get_data()
