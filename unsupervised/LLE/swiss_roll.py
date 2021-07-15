import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sklearn
import sklearn.datasets


def LLE(X, k):
    N = len(X)
    W = np.zeros([N, N])

    for i in range(N):
        d = len(X[i])
        # find the k nearest neighbours of the ith point
        distances = np.linalg.norm(X-X[i].reshape(1, -1), axis=1)
        NN = np.argsort(distances)[1:k+1]
        # compute the weight vector w
        C = np.dot(X[i]-X[NN], (X[i]-X[NN]).T)
        if k > d:
            C += 1e-3*np.trace(C)*np.eye(k)
        C_inv = np.linalg.inv(C)
        w = np.sum(C_inv, axis=1)/np.sum(C_inv)
        W[i, NN] = w

    M = np.identity(N) - W - W.T + np.dot(W.T, W)
    E = np.linalg.svd(M)[0][:, -3:-1]

    return E


if __name__ == "__main__":

    X, T = sklearn.datasets.make_swiss_roll(n_samples=1000, noise=0.25)
    plt.figure(figsize=(5, 5))
    ax = plt.gca(projection='3d')
    ax.view_init(elev=10., azim=105)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=T)
    plt.savefig('./figs/swiss_roll.png')

    f = plt.figure(figsize=(12, 3))
    for t, (k, noise) in enumerate([(2, 0.1), (10, 0.1), (25, 0.1), (10, 1)]):
        X, T = sklearn.datasets.make_swiss_roll(n_samples=1000, noise=noise)

        embedding = LLE(X, k=k)
        ax = f.add_subplot(1, 4, t+1)
        ax.set_title('k=%d, noise=%.1f' % (k, noise))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(embedding[:, 0], embedding[:, 1], c=T)
    plt.savefig('./figs/embeddings.png')
