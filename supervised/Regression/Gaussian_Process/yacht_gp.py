import utils
import datasets
import numpy as np
import matplotlib.pyplot as plt

from gp_regressor import GP_Regressor


if __name__ == '__main__':
    Xtrain, Ytrain, Xtest, Ytest = utils.split(*datasets.yacht())
    xmean, xstd = Xtrain.mean(axis=0), Xtrain.std(axis=0)
    ymean, ystd = Ytrain.mean(), Ytrain.std()

    Xtrain_std = (Xtrain-xmean)/xstd
    Ytrain_std = (Ytrain-ymean)/ystd

    Xtest_std = (Xtest-xmean)/xstd
    Ytest_std = (Ytest-ymean)/ystd

    width = np.linspace(0.05, 2.0, num=50)
    noise = np.linspace(0.005, 0.040, num=50)
    W, N = np.meshgrid(width, noise)
    lltrain = np.zeros((noise.shape[0], width.shape[0]))
    lltest = np.zeros((noise.shape[0], width.shape[0]))
    for i in range(len(noise)):
        for j in range(len(width)):
            gp = GP_Regressor(
                Xtrain_std, Ytrain_std, width[j], noise[i])
            lltrain[i, j] = gp.loglikelihood(Xtrain_std, Ytrain_std)
            lltest[i, j] = gp.loglikelihood(Xtest_std, Ytest_std)
    m = 50
    M = max(lltrain.max(), lltest.max())
    f=plt.figure(figsize=(12,6))

    p=f.add_subplot(1,2,1)
    p.set_title('log P(train | GP posterior)')
    p.set_xlabel('width')
    p.set_ylabel('noise')
    CS=plt.contour(W, N, lltrain, levels=np.arange(m, M, 20))
    p.clabel(CS,inline=1,fontsize=10)

    p=f.add_subplot(1,2,2)
    p.set_title('log P(test | GP posterior)')
    p.set_xlabel('width')
    p.set_ylabel('noise')
    CS=plt.contour(W, N, lltest, levels=np.arange(m, M, 20))
    p.clabel(CS,inline=1,fontsize=10)
    plt.savefig('./figs/yacht_dataset.png')





