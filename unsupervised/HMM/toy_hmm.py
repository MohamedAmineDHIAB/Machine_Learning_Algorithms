import numpy
import hmm


def pprint(L:list):
    s=''
    for i in range(len(L)):

        if i%20==0 and i>0:
            s+='\n'
        s+=str(L[i])+','
    print(s)


if __name__ == '__main__':

    O = numpy.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                    0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                    0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                    0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                    1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                    0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0])

    hmmtoy = hmm.HMM(4, 2)

    for k in range(100):
        hmmtoy.loaddata(O)
        hmmtoy.forward()
        hmmtoy.backward()
        hmmtoy.learn()
    print('\ngiven sequence:\n')
    pprint(list(O))
    print('\n',)
    print('A')
    print("\n".join([" ".join(['%.3f' % a for a in aa]) for aa in hmmtoy.A]))
    print(' ')
    print('B')
    print("\n".join([" ".join(['%.3f' % b for b in bb]) for bb in hmmtoy.B]))
    print(' ')
    print('Pi')
    print("\n".join(['%.3f' % b for b in hmmtoy.Pi])+"\n")
