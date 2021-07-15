import numpy as np
from anomalymodel import AnomalyModel
import utils
import scipy


class KDE(AnomalyModel):

    def __init__(self, gamma):
        self.gamma = gamma

    def fit(self, X):
        self.X = X

    def energy(self, X):
        '''
        X_train = self.X
        X_test = X
        X_train = X_train[:, None, :]
        X_test = X_test[None, :, :]

        Y = np.linalg.norm(X_train-X_test, axis=2)**2
        epsilon = 1e-30
        E = -np.log(np.sum(np.exp(-self.gamma*Y), axis=0)+epsilon)
        '''
        D = scipy.spatial.distance.cdist(self.X, X, 'sqeuclidean')
        E = -scipy.special.logsumexp(-self.gamma*D, axis=0)
        return E


if __name__ == '__main__':
    Xr, Xi, Xo = utils.getdata()
    for gamma in np.logspace(-2, 0, 10):

        kde = KDE(gamma)
        kde.fit(Xr)
        print('gamma = %5.3f  AUROC = %5.3f' % (gamma, kde.auroc()))
