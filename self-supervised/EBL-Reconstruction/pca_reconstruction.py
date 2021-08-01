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


def pca(z, x, d):
    pca = PCA(n_components=d)
    pca.fit(x.numpy())
    y = pca.transform(z.numpy())
    y = pca.inverse_transform(y)
    y = torch.from_numpy(y)
    return y


if __name__ == '__main__':
    Xr, Xt = utils.getdata()
    xmask = removepatch(Xt[:10])[0]
    utils.vis10(xmask)

    Xn, m = removepatch(Xt[:10])

    utils.vis10(pca(Xn, Xr, 10)*m+Xn)
    plt.savefig("./figs/pca_10pc.png")
    utils.vis10(pca(Xn, Xr, 60)*m+Xn)
    plt.savefig("./figs/pca_60pc.png")
    utils.vis10(pca(Xn, Xr, 360)*m+Xn)
    plt.savefig("./figs/pca_360pc.png")
