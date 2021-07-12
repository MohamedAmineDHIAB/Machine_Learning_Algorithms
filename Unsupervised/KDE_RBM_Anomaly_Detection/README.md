# Energy Based Models:

##  Kernel Density Estimation (KDE):

The energy function for KDE model is defined as follows :

<img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\LARGE&space;\centering&space;E(X)=-log(\sum_{i=1}^{N}{exp(-\gamma\lVert&space;x-x_i&space;\rVert^2)})" title="\LARGE \centering E(X)=-log(\sum_{i=1}^{N}{exp(-\gamma\lVert x-x_i \rVert^2)})" />


##  Restricted Boltzmann Machine (RBM) :

The energy function for RBM model composed of 100 binary hidden units is defined as follows :

<img src="https://latex.codecogs.com/png.latex?\dpi{80}&space;\LARGE&space;\boldsymbol{h}&space;\in&space;\{0,1\}^{100}" title="\LARGE \boldsymbol{h} \in \{0,1\}^{100}" /> :

<img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\LARGE&space;\centering&space;E(\boldsymbol{x},\boldsymbol{h})&space;=&space;-\boldsymbol{x}^\top&space;\boldsymbol{a}&space;-&space;\boldsymbol{x}^\top&space;W&space;\boldsymbol{h}&space;-&space;\boldsymbol{h}^\top\boldsymbol{b}" title="\LARGE \centering E(\boldsymbol{x},\boldsymbol{h}) = -\boldsymbol{x}^\top \boldsymbol{a} - \boldsymbol{x}^\top W \boldsymbol{h} - \boldsymbol{h}^\top\boldsymbol{b}" />
