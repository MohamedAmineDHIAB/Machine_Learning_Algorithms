import utils
import datasets
import numpy
import matplotlib.pyplot as plt

from gp_regressor import GP_Regressor


if __name__ == '__main__':

    # Open the toy data
    Xtrain, Ytrain, Xtest, Ytest = utils.split(*datasets.toy())

    # Create an analysis distribution
    Xrange = numpy.arange(-3.5, 3.51, 0.025)[:, numpy.newaxis]

    f = plt.figure(figsize=(18, 15))

    # Loop over several parameters:
    for i, noise in enumerate([2.5, 0.5, 0.1]):
        for j, width in enumerate([0.1, 0.5, 2.5]):

            # Create Gaussian process regressor object
            gp = GP_Regressor(Xtrain, Ytrain, width, noise)

            # Compute the predicted mean and variance for test data
            mean, cov = gp.predict(Xrange)
            var = cov.diagonal()

            # Compute the log-likelihood of training and test data
            lltrain = gp.loglikelihood(Xtrain, Ytrain)
            lltest = gp.loglikelihood(Xtest, Ytest)

            # Plot the data
            p = f.add_subplot(3, 3, 3*i+j+1)
            p.set_title('noise=%.1f width=%.1f lltrain=%.1f, lltest=%.1f' %
                        (noise, width, lltrain, lltest))
            p.set_xlabel('x')
            p.set_ylabel('y')
            p.scatter(Xtrain, Ytrain, color='green',
                      marker='x')  # training data
            p.scatter(Xtest, Ytest, color='green', marker='o')   # test data
            p.plot(Xrange, mean, color='blue')                  # GP mean
            p.plot(Xrange, mean+var**.5, color='red')           # GP mean + std
            p.plot(Xrange, mean-var**.5, color='red')           # GP mean - std
            p.set_xlim(-3.5, 3.5)
            p.set_ylim(-4, 4)
    plt.savefig('./figs/noise_width_effect.png')
