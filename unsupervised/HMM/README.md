# Hidden Markov Model:


Here we will try experiment with hidden Markov models, in particular, applying them to modeling character sequences, and analyzing the learned solution. 
The file `hmm.py` contains basic implementation of an HMM and of the Baum-Welch training algorithm. 
The names of variables used in the code and the references to equations are taken from the HMM paper by Rabiner et al.


### HMM toy example :

We first look at a toy example of an HMM trained on a binary sequence. The training procedure below consists of 100 iterations of the Baum-Welch procedure. It runs the HMM learning algorithm for some toy binary data and prints the parameters learned by the HMM (i.e. matrices `A` and `B`).

![image](https://user-images.githubusercontent.com/85687148/126361341-1f4708b6-c2ae-4d55-825c-e71d6f4bb51e.png)

