import torch
import torch.nn as nn
import utils
import numpy
from matplotlib import pyplot as plt


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
    utils.vis10(xmask)
    plt.show()
