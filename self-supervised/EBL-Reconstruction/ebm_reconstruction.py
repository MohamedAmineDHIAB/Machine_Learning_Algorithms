import torch
import torch.nn as nn
import utils
import numpy
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import torch.optim as optim
from tqdm import tqdm as tq


def removepatch(X):
    mask = torch.zeros(len(X), 28, 28)
    for i in range(len(X)):
        j = numpy.random.randint(-4, 5)
        k = numpy.random.randint(-4, 5)
        mask[i, 11+j:17+j, 11+k:17+k] = 1
    mask = mask.view(len(X), 784)
    return (X*(1-mask)).data, mask


def train(enet, gnet, epochs=100, lrate=0.05, mini_batch=100):

    N = 10000

    optimizer = optim.SGD(list(enet.parameters()) +
                          list(gnet.parameters()), lr=lrate)

    for epoch in tq(range(epochs+1)):

        for i in range(N//mini_batch):

            optimizer.zero_grad()

            # Take a minibatch and train it
            x = Xr[mini_batch*i:mini_batch*(i+1)].data*1.0
            z, m = removepatch(x)

            # Build the forward pass from the input until the loss function

            xgen = gnet(x)
            xgen = (2*xgen).data-xgen
            xgen = x*(1-m)+xgen*m
            edata = enet(x)
            egen = enet(xgen)

            err = torch.log(1+torch.exp(edata-egen)).mean()

            # Compute the gradient and perform one step of gradient descent
            err.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(epoch, err)


if __name__ == '__main__':
    Xr, Xt = utils.getdata()
    xmask = removepatch(Xt[:10])[0]
    torch.manual_seed(0)
    # build energy model
    enet = nn.Sequential(
        nn.Linear(784, 256), nn.Hardtanh(),
        nn.Linear(256, 256), nn.Hardtanh(),
        nn.Linear(256, 1),
    )
    # build generative model
    gnet = nn.Sequential(
        nn.Linear(784, 256), nn.Hardtanh(),
        nn.Linear(256, 256), nn.Hardtanh(),
        nn.Linear(256, 784), nn.Hardtanh()
    )
    train(enet, gnet, epochs=50, lrate=0.08, mini_batch=1000)

    x = Xt[:10]
    z, m = removepatch(x)
    utils.vis10(z)
    plt.savefig('./figs/befoe_ebm.png')
    utils.vis10(gnet(z)*m+z*(1-m))
    plt.savefig('./figs/after_ebm.png')
