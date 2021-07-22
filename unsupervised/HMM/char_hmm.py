import numpy

from hmm import HMM
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

na = numpy.newaxis

# Download a subset of the newsgroup dataset
newsgroups_train = fetch_20newsgroups(subset='train', categories=['sci.med'])
newsgroups_test = fetch_20newsgroups(subset='test', categories=['sci.med'])

# Sample a sequence of T characters from the dataset
# that the HMM can read (0=whitespace 1-26=A-Z).
#
# Example of execution:
# O = sample(newsgroups_train.data)
# O = sample(newsgroups_test.data)
#


def sample(data, T=50):
    i = numpy.random.randint(len(data))
    O = data[i].upper().replace('\n', ' ')
    O = numpy.array([ord(s) for s in O])
    O = numpy.maximum(O[(O >= 65)*(O < 90)+(O == 32)]-64, 0)

    j = numpy.random.randint(len(O)-T)
    return O[j:j+T]

# Takes a sequence of integers between 0 and 26 (HMM representation)
# and converts it back to a string of characters


def tochar(O):
    return "".join(["%s" % chr(o) for o in (O+32*(O == 0)+64*(O > 0.5))])

# Char Hidden Markov Model class


class HMMChar(HMM):

    # Baum-Welch parameter update (Eq. 36-40)
    def learn(self):

        # Compute gamma
        self.gamma = self.alpha*self.beta / self.pobs

        # Compute xi and psi
        self.xi = self.alpha[:-1, :, na]*self.A[na, :, :] * \
            self.beta[1:, na, :]*self.Z[1:, na, :] / self.pobs
        self.psi = self.gamma[:, :, na] * \
            (self.O[:, na, na] == numpy.arange(self.B.shape[1])[na, na, :])

        # Update HMM parameters
        self.A = self.A*0.9+0.1 * \
            (self.xi.sum(axis=0) / self.gamma[:-1].sum(axis=0)[:, na])
        self.B = self.B*0.9+0.1 * \
            (self.psi.sum(axis=0) / self.gamma.sum(axis=0)[:, na])
        self.Pi = self.Pi*0.9+0.1*(self.gamma[0])

    def generate(self, T):
        nbr_states, nbr_obs = self.B.shape[0], self.B.shape[1]
        state = numpy.random.choice(nbr_states, p=self.Pi)
        obs = []
        for _ in range(T):
            obs.append(numpy.random.choice(nbr_obs, p=self.B[state]))
            state = numpy.random.choice(nbr_states, p=self.A[state])
        return(numpy.array(obs))


if __name__ == '__main__':

    def trainsample(): return sample(newsgroups_train.data, 100)
    def testsample(): return sample(newsgroups_test.data, 100)

    hmmchar = HMMChar(300, 27)
    pobstest = []
    pobstrain = []

    print('\nLaunching training of HMM model ...')
    print('-'*100)

    for k in range(3000):

        train_batch = trainsample()
        test_batch = testsample()
        hmmchar.loaddata(train_batch)
        hmmchar.forward()
        hmmchar.backward()
        hmmchar.learn()

        hmmchar.loaddata(train_batch)
        hmmchar.forward()
        pobstrain.append(hmmchar.pobs)

        hmmchar.loaddata(test_batch)
        hmmchar.forward()
        pobstest.append(hmmchar.pobs)

        if k % 1000 == 0:
            print(
                f'iteration={k} logptrain={numpy.mean(numpy.log(pobstrain)):.2f} logptest={numpy.mean(numpy.log(pobstest)):.2f}')
    print('-'*100)
    print('Testing the HMM model...')
    print('-'*100)
    print("original:\n"+tochar(sample(newsgroups_test.data, T=100)))
    print("\nlearned:\n"+tochar(hmmchar.generate(100)))
    print("\nrandom:\n" + tochar(HMMChar(300, 27).generate(100))+'\n')
