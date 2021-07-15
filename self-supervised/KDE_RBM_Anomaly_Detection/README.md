# Energy Based Models:

##  Kernel Density Estimation (KDE):

The energy function for KDE model is defined as follows :

<img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\LARGE&space;\centering&space;E(X)=-log(\sum_{i=1}^{N}{exp(-\gamma\lVert&space;x-x_i&space;\rVert^2)})" title="\LARGE \centering E(X)=-log(\sum_{i=1}^{N}{exp(-\gamma\lVert x-x_i \rVert^2)})" />


##  Restricted Boltzmann Machine (RBM) :

The energy function for RBM model composed of 100 binary hidden units is defined as follows :

<img src="https://latex.codecogs.com/png.latex?\dpi{80}&space;\LARGE&space;\boldsymbol{h}&space;\in&space;\{0,1\}^{100}" title="\LARGE \boldsymbol{h} \in \{0,1\}^{100}" /> :

<img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\LARGE&space;\centering&space;E(\boldsymbol{x},\boldsymbol{h})&space;=&space;-\boldsymbol{x}^\top&space;\boldsymbol{a}&space;-&space;\boldsymbol{x}^\top&space;W&space;\boldsymbol{h}&space;-&space;\boldsymbol{h}^\top\boldsymbol{b}" title="\LARGE \centering E(\boldsymbol{x},\boldsymbol{h}) = -\boldsymbol{x}^\top \boldsymbol{a} - \boldsymbol{x}^\top W \boldsymbol{h} - \boldsymbol{h}^\top\boldsymbol{b}" />

We consider the MNIST dataset and define the class "0" to be normal (inlier) and the remain classes (1-9) to be anomalous (outlier). We consider that we have a training set `Xr` composed of 100 normal data points. The variables `Xi` and `Xo` denote normal and anomalous test data.

The 100 training points are visualized below:
<img src="https://raw.githubusercontent.com/MohamedAmineDHIAB/Machine_Learning_Algorithms/main/self-supervised/KDE_RBM_Anomaly_Detection/data/100_train_0.png" title="100 points with target=0" />

The receptive field connecting the input image to a particular hidden unit of the RBM :

<img src="https://raw.githubusercontent.com/MohamedAmineDHIAB/Machine_Learning_Algorithms/main/Unsupervised/KDE_RBM_Anomaly_Detection/data/rbm_learned_params.png" title="Weights of the RBM"/>


