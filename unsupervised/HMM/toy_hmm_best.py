import numpy
import hmm


def pprint(L: list):
    s = ''
    for i in range(len(L)):

        if i % 20 == 0 and i > 0:
            s += '\n'
        s += str(L[i])+','
    print(s)


if __name__ == '__main__':

    O = numpy.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                    0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                    0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                    0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                    1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                    0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0])

    for N in [2, 4, 8, 16]:
        print('\nN=%d' % N)
        epsilon = 1e-100
        for i in range(4):

            hmmtoy = hmm.HMM(N, 2)

            Otrain = O[:len(O)//2]
            Otest = O[len(O)//2:]

            for k in range(100):
                hmmtoy.loaddata(Otrain)
                hmmtoy.forward()
                hmmtoy.backward()
                hmmtoy.learn()

            hmmtoy.loaddata(Otrain)
            hmmtoy.forward()
            lptrain = numpy.log(hmmtoy.pobs+epsilon)

            hmmtoy.loaddata(Otest)
            hmmtoy.forward()
            lptest = numpy.log(hmmtoy.pobs+epsilon)

            print(f'trail={i+1} lptrain={lptrain:.2f} lptest={lptest:.2f}')
        print('\n')
