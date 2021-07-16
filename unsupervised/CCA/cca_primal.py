import numpy as np
from scipy.linalg import eigh
from matplotlib import pyplot as plt
import utils


def CCAprimal(X, Y):

    # means
    X = X-np.mean(X, axis=0)
    Y = Y-np.mean(Y, axis=0)
    # shapes
    N = X.shape[0]
    d1 = X.shape[1]
    d2 = Y.shape[1]
    # matrices
    C_XX = X.T@X/N
    C_YY = Y.T@Y/N
    C_XY = X.T@Y/N
    C_YX = Y.T@X/N
    Z1 = np.zeros((d1, d1))
    Z2 = np.zeros((d2, d2))
    A = np.concatenate((np.concatenate((Z1, C_XY), axis=1),
                       np.concatenate((C_YX, Z2), axis=1)), axis=0)
    B = np.concatenate((np.concatenate((C_XX, Z2), axis=1),
                       np.concatenate((Z1, C_YY), axis=1)), axis=0)
    # resolving the eigval problem
    eigvals, eigvecs = eigh(A, B, eigvals_only=False)
    max_index = np.argmax(eigvals)
    wx, wy = -eigvecs[:, max_index][:d1], -eigvecs[:, max_index][d1:]

    return wx, wy


if __name__ == '__main__':
    X, Y = utils.getdata()
    p1, p2 = utils.plotdata(X, Y)
    wx, wy = CCAprimal(X, Y)
    plt.savefig('./figs/primal_data.png')
    p1.arrow(0, 0, 1*wx[0], 1*wx[1], color='red', width=0.1)
    p2.arrow(0, 0, 1*wy[0], 1*wy[1], color='red', width=0.1)
    plt.savefig('./figs/primal_eigenvectors.png')
    plt.figure(figsize=(6, 2))
    plt.plot(np.dot(X, wx))
    plt.plot(np.dot(Y, wy))
    plt.savefig('./figs/primal_projections.png')
