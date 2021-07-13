import numpy as np
from anomalymodel import AnomalyModel
import utils
from matplotlib import pyplot as plt


def sigm(t): return np.tanh(0.5*t)*0.5+0.5
def realize(t): return 1.0*(t > np.random.uniform(0, 1, t.shape))


class RBM(AnomalyModel):

    def __init__(self, X, h):
        self.mb = X.shape[0]
        self.d = X.shape[1]
        self.h = h
        self.lr = 0.1

        # Model parameters
        self.A = np.zeros([self.d])
        self.W = np.random.normal(
            0, self.d**-.25 * self.h**-.25, [self.d, self.h])
        self.B = np.zeros([self.h])

    def fit(self, X, verbose=False):

        Xm = np.zeros([self.mb, self.d])

        for i in np.arange(1001):

            # Gibbs sampling (PCD)
            Xd = X*1.0
            Zd = realize(sigm(Xd.dot(self.W)+self.B))
            Zm = realize(sigm(Xm.dot(self.W)+self.B))
            Xm = realize(sigm(Zm.dot(self.W.T)+self.A))

            # Update parameters
            self.W += self.lr * \
                ((Xd.T.dot(Zd) - Xm.T.dot(Zm)) / self.mb - 0.01*self.W)
            self.B += self.lr*(Zd.mean(axis=0)-Zm.mean(axis=0))
            self.A += self.lr*(Xd.mean(axis=0)-Xm.mean(axis=0))

            if verbose:
                # print AUROC every 100 iterations
                if i % 100 == 0:
                    print('it= %5.0f AUROC = %5.3f' % (i, self.auroc()))

    def energy(self, X):

        # RBM energy
        E = -X@self.A.T-np.sum(np.log(1+np.exp(X@self.W+self.B)), axis=-1)
        return E


if __name__ == 'main':
    Xr, Xi, Xo = utils.getdata()
    rbm = RBM(Xr, 100)
    rbm.fit(Xr, verbose=True)
    plt.figure(figsize=(16, 4))
    plt.imshow(Xr.reshape(5, 20, 28, 28).transpose(
        0, 2, 1, 3).reshape(140, 560))
    plt.show()
    plt.figure(figsize=(16, 4))
    plt.imshow(rbm.W.T.reshape(5, 20, 28, 28).transpose(
        0, 2, 1, 3).reshape(140, 560))
    plt.show()
