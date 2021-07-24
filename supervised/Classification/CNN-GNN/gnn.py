import torch
import torch.nn as nn
from numpy import newaxis as na
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
        '''
        D = np.sum(A, axis=2)
        lamda = A/((D[:, na, :]**0.5)*(D[:, :, na]**0.5)+1e-10)
        '''
        D = A.sum(dim=2)
        lamda = A/(D.view(len(A), 1, -1)**.5 * D.view(len(A), -1, 1)**.5+1e-6)
        H1 = lamda.matmul(self.U).matmul(self.W)
        H1 = H1*(H1 > 0)
        H2 = lamda.matmul(H1).matmul(self.W)
        H2 = H2*(H2 > 0)
        H3 = lamda.matmul(H2).matmul(self.W)
        H3 = H3*(H3 > 0)

        Y = H3.matmul(self.V).sum(dim=1)

        return (Y)


if __name__ == '__main__':
    Ar, Tr, At, Tt = utils.graphdata()
    torch.manual_seed(0)
    dnn = utils.NNClassifier(nn.Sequential(
        nn.Linear(225, 512), nn.ReLU(), nn.Linear(512, 3)), flat=True)
    torch.manual_seed(0)
    gnn = utils.NNClassifier(GNN(15, 25, 3))

    for name, net in [('DNN', dnn), ('GNN', gnn)]:
        net.fit(Ar, Tr, lr=0.01, epochs=500)
        Yr = net.predict(Ar)
        Yt = net.predict(At)
        acctr = (Yr.max(dim=1)[1] == Tr).data.numpy().mean()
        acctt = (Yt.max(dim=1)[1] == Tt).data.numpy().mean()
        print('-'*50)
        print('name: %10s  train: %.3f  test: %.3f' % (name, acctr, acctt))
    print('-'*50)
