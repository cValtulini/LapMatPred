################################################################################
# PACKAGES
################################################################################
import numpy as np

from LapMatPred.myUtilities import make_symmetric, remove_diag
from matplotlib import pyplot as plt
from pygsp import graphs

################################################################################
# FUNCTIONS
################################################################################
def erdos_renyi(n, prob, directed=False, rng=None):
    """
    erdos_renyi return the matrix of weights (adjacency) for a graph with n
    nodes and p probability of connection between two nodes. Weights in the adjacency
    matrix are extracted from a uniform distribution between [0, 1]

    Parameters
    ----------
    n: int
        The number of nodes in the graph
    prob: float
        The probability of connection between every two nodes
    directed: bool, default to False
        If True, the edges matrix is returned as is, removing diagonal entries. If
        False we take the triangular upper matrix and make the edges matrix symmetric
        before removing the diagonal and returning it
    rng: numpy.random.Generator, default to None
        The random number generator to use

    Returns
    -------
    numpy.ndarray, shape=(n, n)
        The adjacency matrix of the graph
    """
    if rng is None:
        rng = np.random.default_rng()

    e = rng.choice([0, 1], size=(n, n), p=(1 - prob, prob))  # edge matrix
    a = e * rng.uniform(size=e.shape)  # adjacency matrix

    if directed:
        return remove_diag(a)

    return remove_diag(make_symmetric(a))


def graph_laplacian(a):
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


def plot_graph(w, title=None):
    """
    Simple function to plot a graph given its weight matrix

    Parameters
    ----------
    w: numpy.ndarray, shape=(n, n)
        The weight matrix of the graph
    title: str, default to None
        The title for the plot

    Returns
    -------
    None
    """
    graph = graphs.Graph(w)
    graph.set_coordinates()

    graph.plot(backend="matplotlib", title=title, figsize=(10, 10))

    plt.show()


def reconstruct_laplacian(x, original_shape):
    """
    reconstructLaplacian reconstructs the laplacian matrix assuming it has been
    decomposed extracting its upper triangolar without the main diagonal and
    flattening it. If original_shape is a Tuple[int, int, int] assumes that x has
    shape (n_samples, n).

    Parameters
    ----------
    x: numpy.ndarray, shape=(n, )
        The flattened upper triangular part of the laplacian matrix
    original_shape: tuple
        The original shape of the laplacian matrix, helps avoiding inference of
        the shape from the flattened array

    Returns
    -------

    """
    new = np.zeros(shape=original_shape)
    ind_i, ind_j = np.triu_indices_from(
        # if new.ndim == 3 we are assuming its a batch of matrices that need to
        # be reeconstructed
        np.zeros(shape=(original_shape[-2], original_shape[-1])),
        k=1,
    )

    if new.ndim == 2:
        for i, j, element in zip(ind_i, ind_j, x):
            new[i, j] = -element
            new[j, i] = -element

        for i in range(original_shape[-1]):
            new[i, i] = -np.sum(new[i, :])
    elif new.ndim == 3:
        for t in range(original_shape[0]):
            for i, j, element in zip(ind_i, ind_j, x[t]):
                new[t, i, j] = -element
                new[t, j, i] = -element

            for i in range(original_shape[-1]):
                new[t, i, i] = -np.sum(new[t, i, :])

    return new
