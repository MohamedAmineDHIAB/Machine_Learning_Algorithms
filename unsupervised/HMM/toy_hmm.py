import numpy,hmm


if __name__ == '__main__':

    O = numpy.array([1,0,1,0,1,1,0,0,1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,1,1,0,0,1,1,
                    0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,
                    0,0,1,0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,
                    0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,1,
                    1,0,0,0,1,1,0,0,1,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,
                    0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,
                    0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0])

    hmmtoy = hmm.HMM(4,2)

    for k in range(100):
        hmmtoy.loaddata(O)
        hmmtoy.forward()
        hmmtoy.backward()
        hmmtoy.learn()

    print('A')
    print("\n".join([" ".join(['%.3f'%a for a in aa]) for aa in hmmtoy.A]))
    print(' ')
    print('B')
    print("\n".join([" ".join(['%.3f'%b for b in bb]) for bb in hmmtoy.B]))
    print(' ')
    print('Pi')
    print("\n".join(['%.3f'%b for b in hmmtoy.Pi]))