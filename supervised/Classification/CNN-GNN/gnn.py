import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import utils
import numpy
import numpy as np
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
        D = np.sum(A, axis=2)
        lamda = D**0.5@A@D**0.5
        H1 = lamda@self.U@self.W*((lamda@self.U@self.W) > 0)
        H2 = lamda@H1@self.W*((lamda@H1@self.W) > 0)
        H3 = lamda@H2@self.W*((lamda@H2@self.W) > 0)
        Y = np.ones((1, A.shape[0]))@H3@self.V

        return (Y)


if __name__ == '__main__':
    Ar,Tr,At,Tt = utils.graphdata()

    torch.manual_seed(0)
    dnn = utils.NNClassifier(nn.Sequential(nn.Linear( 225,512), nn.ReLU(),nn.Linear(512,3)),flat=True)
    torch.manual_seed(0)
    gnn = utils.NNClassifier(GNN(15,25,3))

    for name,net in [('DNN',dnn),('GNN',gnn)]:
        net.fit(Ar,Tr,lr=0.01,epochs=500)
        Yr = net.predict(Ar)
        Yt = net.predict(At)
        acctr = (Yr.max(dim=1)[1] == Tr).data.numpy().mean()
        acctt = (Yt.max(dim=1)[1] == Tt).data.numpy().mean()
        print('name: %10s  train: %.3f  test: %.3f'%(name,acctr,acctt))
