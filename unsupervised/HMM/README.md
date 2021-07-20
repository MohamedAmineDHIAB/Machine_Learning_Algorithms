# Hidden Markov Model:


Here we will try experiment with hidden Markov models, in particular, applying them to modeling character sequences, and analyzing the learned solution. 
The file `hmm.py` contains basic implementation of an HMM and of the Baum-Welch training algorithm. 
The names of variables used in the code and the references to equations are taken from the HMM paper by Rabiner et al.


### HMM toy example :

We first look at a toy example of an HMM trained on a binary sequence. The training procedure below consists of 100 iterations of the Baum-Welch procedure. It runs the HMM learning algorithm for some toy binary data and prints the parameters learned by the HMM (i.e. matrices `A` and `B`).

![image](https://user-images.githubusercontent.com/85687148/126363255-7eb38d8f-8201-49d1-9021-5cbb8851149e.png)


A : contains the probabilities of transitions from state i to state j , where i,j in [1,4]

B : contains the probabilities of emissiosn (column j is probability of row i emmitting the value j)

Pi : contains the prability of a state being the initial state

=> We can see that the model learned from the random given sequence , a pattern which consists of emitting "1.0.0.0" many times + a noise , which corresponds to the emissions matrix B



