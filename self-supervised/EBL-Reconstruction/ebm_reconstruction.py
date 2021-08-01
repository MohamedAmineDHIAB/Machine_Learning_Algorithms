import torch
import torch.nn as nn
import utils
import numpy
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def removepatch(X):
    mask = torch.zeros(len(X), 28, 28)
    for i in range(len(X)):
        j = numpy.random.randint(-4, 5)
        k = numpy.random.randint(-4, 5)
        mask[i, 11+j:17+j, 11+k:17+k] = 1
    mask = mask.view(len(X), 784)
    return (X*(1-mask)).data, mask


if __name__ == '__main__':
    Xr, Xt = utils.getdata()
    xmask = removepatch(Xt[:10])[0]
    torch.manual_seed(0)
    enet = nn.Sequential(
        nn.Linear(784, 256), nn.Hardtanh(),
        nn.Linear(256, 256), nn.Hardtanh(),
        nn.Linear(256, 1),
    )

    Xn, m = removepatch(Xt[:10])

    utils.vis10(pca(Xn, Xr, 10)*m+Xn*(1-m))
    utils.vis10(pca(Xn, Xr, 60)*m+Xn*(1-m))
    utils.vis10(pca(Xn, Xr, 360)*m+Xn*(1-m))

    plt.show()
