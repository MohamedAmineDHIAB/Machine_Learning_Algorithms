import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def pca(X, n):
    '''
    input : X : array of data points
            n : number of the desired principal components
    output : nPC : the first n principal components of the data points
    '''

    # mean Centering the data
    X_meaned = X - np.mean(X, axis=0)
    # calculating the covariance matrix of the mean-centered data
    cov_mat = np.cov(X_meaned, rowvar=False)
    # Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    # similarly sort the eigenvectors
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # select the first n eigenvectors
    n_components = n
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    nPC = eigenvector_subset
    X_reduced = X_meaned@nPC
    return(X_reduced)


if __name__ == '__main__':

    # Get the IRIS dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(url, names=[
                       'sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

    # prepare the data
    x = data.iloc[:, 0:4]

    # prepare the target
    target = data.iloc[:, 4]

    # Get the new embedded data
    x_reduced = pca(x, 2)

    # Creating a Pandas DataFrame of embedded Dataset
    reduced_df = pd.DataFrame(
        x_reduced, columns=['Principal Comp 1', 'Principal Comp 2'])

    # Concat it with target variable to create a complete Dataset
    reduced_df = pd.concat([reduced_df, pd.DataFrame(target)], axis=1)
    plt.figure(figsize=(6, 6))
    sb.scatterplot(data=reduced_df, x='Principal Comp 1', y='Principal Comp 2',
                   hue='target', s=60, palette='rocket')
