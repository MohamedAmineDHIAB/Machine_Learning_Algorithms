import utils
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import eigh


def CCAdual(X, Y):

    # normalization
    X = X-np.mean(X, axis=0)
    Y = Y-np.mean(Y, axis=0)
    # shapes
    N = X.shape[0]
    d1 = X.shape[1]
    d2 = Y.shape[1]
    # matrices
    A = X@X.T
    B = Y@Y.T
    Z = np.zeros((N, N))
    U = np.block([[Z, A@B], [B@A, Z]])
    V = np.block([[A@A, Z], [Z, B@B]])
    V += 1e-3*np.trace(V)*np.eye(2*N)
    _, eigvecs = eigh(U, V, eigvals_only=False)
    alpha_x = eigvecs[:N, -1]
    alpha_y = eigvecs[N:, -1]

    return X.T@alpha_x, Y.T@alpha_y


if __name__ == '__main__':
    X, Y = utils.getHDdata()

    utils.plotHDdata(X[0], Y[0])
    plt.savefig('./figs/dual_data.png')

    wx, wy = CCAdual(X[:100], Y[:100])

    utils.plotHDdata(wx, wy)
    plt.savefig('./figs/dual_eigenvectors.png')

    plt.figure(figsize=(6, 2))
    plt.plot(np.dot(X[100:], wx))
    plt.plot(np.dot(Y[100:], wy))
    plt.savefig('./figs/dual_projections.png')
