import utils
import numpy as np


class GP_Regressor():
    def __init__(self, Xtrain, Ytrain, width, noise):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.width = width
        self.noise = noise

    def predict(self, Xtest):
        Id = np.identity(self.Xtrain.shape[0])*self.noise**2
        sigma = utils.gaussianKernel(self.Xtrain, self.Xtrain, self.width)+Id

        sigma_1 = utils.gaussianKernel(self.Xtrain, Xtest, self.width)

        Id_2 = np.identity(Xtest.shape[0])*self.noise**2
        sigma_2 = utils.gaussianKernel(Xtest, Xtest, self.width)+Id_2


        sigma_inv = np.linalg.inv(sigma)

        c_star = sigma_2-(sigma_1.T@(sigma_inv@sigma_1))
        mu_star = sigma_1.T@sigma_inv@self.Ytrain

        return(mu_star, c_star)

    def loglikelihood(self, Xtest, Ytest):

        mu_star, c_star = self.predict(Xtest)

        A = (Ytest-mu_star).T@(np.linalg.inv(c_star)@(Ytest-mu_star))
        B = np.linalg.slogdet(c_star)[1]

        log_p = -0.5*(A+B+self.Ytrain.shape[0]*np.log(2*np.pi))

        return(log_p)
