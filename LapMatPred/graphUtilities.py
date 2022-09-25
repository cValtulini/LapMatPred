#########################################################################################
# PACKAGES
#########################################################################################
import numpy as np

from LapMatPred.myUtilities import makeSymmetric, removeDiag
from matplotlib import pyplot as plt
from pygsp import graphs, plotting

#########################################################################################
# FUNCTIONS
#########################################################################################
def erdosRenyi(n, prob, directed=False):
    """
    erdosRenyi return the matrix of weights (adjacency) for a graph with n
    nodes and p probability of connection between two nodes. Weights in the adjacency
    matrix are extracted from a uniform distribution between [0, 1]

    Parameters
    ----------
    n: int
        The number of nodes in the graph
    prob: int
        The probability of connection between every two nodes
    directed: bool, default to False
        If True, the edges matrix is returned as is, removing diagonal entries. If
        False we take the triangular upper matrix and make the edges matrix symmetric
        before removing the diagonal and returning it

    Returns
    -------
    numpy.ndarray, shape=(n, n)
        The adjacency matrix of the graph
    """
    e = np.random.choice([0, 1], size=(n, n), p=(1-prob, prob))  # edge matrix
    a = e * np.random.uniform(size=e.shape)  # adjacency matrix

    if directed:
        return removeDiag(a)

    return removeDiag(makeSymmetric(a))


def graphLaplacian(a):
    """
    graphMatrices return the laplacian matrix of a graph given its
    adjacency matrix.

    Parameters
    ----------
    a: numpy.ndarray, shape=(n, n)
        The adjacency matrix of the graph

    Returns
    -------

    l: numpy.ndarray, shape=(n, n)
        The laplacian matrix of the graph
    """
    return np.diag(np.sum(a, axis=-1)) - a


def plotGraph(w, title=None):
    graph = graphs.Graph(w)
    graph.set_coordinates()

    graph.plot(backend='matplotlib', title=title, figsize=(10, 10))

    plt.show()