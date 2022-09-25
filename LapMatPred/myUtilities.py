#########################################################################################
# PACKAGES
#########################################################################################
import numpy as np
from matplotlib import pyplot as plt

#########################################################################################
# FUNCTIONS
#########################################################################################
def removeDiag(m):
    """
    removeDiag returns a matrix containing 0 on the diagonal and filled with the values of
    m on the rest of the matrix.

    Parameters
    ----------
    m: numpy.ndarray, shape=(n, n)
        A matrix

    Returns
    -------
    numpy.ndarray, shape=(n, n)
        Matrix m with 0 diagonal values
    """
    return m - np.diag(np.diag(m))


def makeSymmetric(m):
    """
    makeSymmetric creates a symmetric matrix from a matrix m. The returned matrix
    contains the values of the upper triangle of m, both on the upper triangle and on the
    lower triangle, the values on the diagonal are maintained.

    Parameters
    ----------
    m: numpy.ndarray, shape=(n, n)
        A matrix

    Returns
    -------
    numpy.ndarray, shape=(n, n)
        np.triu(m) + np.triu(m').T where m' is m - diag(m)
    """
    return np.triu(m) + np.triu(removeDiag(m)).T


def extractBlock(x, index):
    """
    extractBlock returns matrix x without index-th row and column or array x without
    index-th element.

    Parameters
    ----------
    x: numpy.ndarray, shape=(None, n)
        Matrix from which to extract index-th row and column or array from which to
        extract index-th element
    index: int
        The index of row and column to remove or element to remove from array

    Returns
    -------
    numpy.ndarray, shape=(None, n-1)
        A matrix of dimensions (n-1, n-1) or an array of dimensions n-1
    """
    if x.ndim == 1:
        if index == 0:
            return x[1:]
        elif index == x.shape[0] - 1:
            return x[:-1]
        else:
            return np.hstack((x[:index], x[index + 1:]))
    elif x.ndim == 2:
        if index == 0:
            return x[1:, 1:]
        elif index == x.shape[0] - 1:
            return x[:-1, :-1]
        else:
            _ = np.vstack((x[:index], x[index + 1:]))
            return np.hstack((_[:, :index], _[:, index+1:]))


def updateBlockMatrix(x_block, x_col, x_scal, index):
    """
    updateBlockMatrix returns a matrix built from x_block, x_col and x_scal by inserting
    x_col at index-th row and column and x_scal at (index-th, index-th) position of the
    new matrix.

    Parameters
    ----------
    x_block: numpy.ndarray, shape=(n, n)
        A matrix
    x_col: numpy.ndarray, shape=(n, 1)
        A column vector
    x_scal: float
        A scalar
    index:  int
        The index of row and column where to insert x_col and x_scal at

    Returns
    -------
    numpy.ndarray, shape=(n + 1, n + 1)
        A matrix of dimensions (n + 1, n + 1) obtained inserting x_col as the index-th row
        and column and x_scal as the (index-th, index-th) element, x_col is split at its
        index-th element in order to insert x_scal.
    """
    x = np.zeros((x_block.shape[0] + 1, x_block.shape[1] + 1))

    if index == 0:
        x[0, 0] = x_scal

        x[0, 1:] = x_col
        x[1:, 0] = x_col

        x[1:, 1:] = x_block
    elif index == x_block.shape[0]:
        x[-1, -1] = x_scal

        x[-1, :-1] = x_col
        x[:-1, -1] = x_col

        x[:-1, :-1] = x_block
    else:
        x[index, index] = x_scal

        x[index, :index] = x_col[:index]
        x[index, index + 1:] = x_col[index:]
        x[:index, index] = x_col[:index]
        x[index + 1:, index] = x_col[index:]

        x[:index, :index] = x_block[:index, :index]
        x[:index, index + 1:] = x_block[:index, index:]
        x[index + 1:, :index] = x_block[index:, :index]
        x[index + 1:, index + 1:] = x_block[index:, index:]

    return x


def plotErrorBars(x, y, x_label='', y_label='', title='', labels=None, save=False):
    """
    plotErrorBars plots the error bars of a matrix x and a vector y.

    Parameters
    ----------
    x: numpy.ndarray, shape=(n, 1)
        A column vector
    y: numpy.ndarray, shape=(None, n, m)
        A matrix
    x_label: str
        The label of the x-axis
    y_label: str
        The label of the y-axis
    labels: list, optional
        A list of strings to label the x-axis and the y-axis
    title: str, optional
        The title of the plot
    save: bool, optional
        If True, the plot is saved as a png file

    Returns
    -------
    None
    """
    y_mean = y.mean(axis=-1)
    y_std = y.std(axis=-1)

    plt.figure(figsize=(10, 10))
    plt.title(title)
    if y.ndim == 3:
        if labels is not None:
            for i in range(y.shape[0]):
                plt.errorbar(
                    x, y_mean[i], yerr=y_std[i], fmt='o--',
                    capthick=1, capsize=4, label=labels[i]
                    )
        else:
            for i in range(y.shape[0]):
                plt.errorbar(
                    x, y_mean[i], yerr=y_std[i], fmt='o--', capthick=1, capsize=4
                    )
    else:
        plt.errorbar(x, y_mean, yerr=y_std, fmt='o--', capthick=1, capsize=4)

    plt.xscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xticks(np.log(x))
    # plt.xlim(np.log(x).min(), np.log(x).max())

    if labels is not None:
        plt.legend()

    plt.show()

    if save:
        plt.savefig(f'{title}.png')
