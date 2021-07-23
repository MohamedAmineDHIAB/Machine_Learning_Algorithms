import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import utils
import numpy

from matplotlib import pyplot as plt


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    Xr, Tr = trainset.data.float().view(-1, 1, 28, 28)/127.5-1, trainset.targets
    Xt, Tt = testset.data.float().view(-1, 1, 28, 28)/127.5-1, testset.targets
    return(Xr, Tr, Xt, Tt)


def build_cnn():
    torch.manual_seed(0)
    cnn = utils.NNClassifier(nn.Sequential(
        nn.Conv2d(1, 8, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 24, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(24, 72, 4), nn.ReLU(),
        nn.Conv2d(72, 10, 1)
    ))
    return(cnn)


def build_linear_net():
    torch.manual_seed(0)
    lin = utils.NNClassifier(nn.Sequential(nn.Linear(784, 10)), flat=True)

    return (lin)


def build_fully_connected():
    torch.manual_seed(0)
    fc = utils.NNClassifier(nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10)
    ), flat=True)
    return(fc)


if __name__ == '__main__':
    Xr, Tr, Xt, Tt = get_data()
    cnn = build_cnn()
    lin = build_linear_net()
    fc = build_fully_connected()
    for name, cl in [('linear', lin), ('full', fc), ('conv', cnn)]:
        cl.fit(Xr, Tr, epochs=5)

        errtr = numpy.mean(cl.predict(
            Xr[:]).numpy().argmax(axis=1) == Tr[:].numpy())
        errtt = numpy.mean(cl.predict(
            Xt[:]).numpy().argmax(axis=1) == Tt[:].numpy())
        print('-'*70)
        print('%10s accuracy on train: %.3f  accuracy on test: %.3f' %
              (name, errtr, errtt))
    print('\n')

    for digits in [highest, lowest]:
        plt.figure(figsize=(8, 3))
        plt.axis('off')
        plt.imshow(digits.numpy().reshape(3, 8, 28, 28).transpose(
            0, 2, 1, 3).reshape(28*3, 28*8), cmap='gray')
        plt.show()
