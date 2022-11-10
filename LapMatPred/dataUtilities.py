#########################################################################################
# PACKAGES
#########################################################################################
from LapMatPred.graphUtilities import *

import numpy as np
import tensorflow as tf


#########################################################################################
# FUNCTIONS
#########################################################################################
def createDataset(rng, nodes_number, connection_probability,
                  number_of_realizations, set_size):
    adjacency = np.array(
        [erdosRenyi(nodes_number, connection_probability, rng=rng) for _ in range(
            set_size)]
        )
    laplacian = np.array([graphLaplacian(a) for a in adjacency])

    Q_true = np.eye(nodes_number) + laplacian

    K_matrix = np.linalg.inv(Q_true)

    mean_vector = np.zeros(nodes_number)
    covariance = np.array(
        [np.cov(
            rng.multivariate_normal(
                mean_vector, K_matrix[i], size=number_of_realizations
                ),
            rowvar=False
            ) for i in range(K_matrix.shape[0])]
        )
    return covariance, Q_true


def splitDataset(x, y, train_size, val_size=None):
    """
    Splits the dataset into training, validation and test sets. If val_size is None
    train is given by samples up to train_size and test is given by remaining samples.
    If val_size is not None, train is given by samples up to train_size, validation is
    given by samples from train_size to train_size + val_size and test is given by
    remaining samples.
    Returns None if x and y have different lengths (first shape) or if train_size is
    greater than the number of samples in the first case or additionally also if
    traub_size + val_size is greater than the number of samples in the second case.

    Parameters
    ----------
    x: numpy.ndarray, shape=(number of samples, None)
        The input data
    y: numpy.ndarray, shape=(number of samples, None)
        The ground truths
    train_size: int
        The number of samples to use for the training set
    val_size:
        The number of samples to use for the validation set

    Returns
    -------
    tuple of numpy.ndarray:
        First split of the input data and ground truths
    tuple of numpy.ndarray:
        Second split of the input data and ground truths
    tuple of numpy.ndarray:
        (Optional) Third split of the input data and ground truths if val_size is not None

    """

    if x.shape[0] != y.shape[0]:
        print('Error: x and y have different number of samples')
        return None
    if val_size is None:
        if x.shape[0] < train_size:
            print('Error: train_size is greater than the number of samples')
            return None

        x_train = tf.convert_to_tensor(x[:train_size])
        x_test = tf.convert_to_tensor(x[train_size:])
        y_train = tf.convert_to_tensor(y[:train_size])
        y_test = tf.convert_to_tensor(y[train_size:])

        return (x_train, y_train), (x_test, y_test)
    else:
        if x.shape[0] < train_size or x.shape[0] < train_size + val_size:
            print(
                'Error: dataset size is not equal to the sum of train and test set sizes'
                )
            return None

        x_train = tf.convert_to_tensor(x[:train_size])
        x_val = tf.convert_to_tensor(x[train_size:train_size+val_size])
        x_test = tf.convert_to_tensor(x[train_size+val_size:])

        y_train = tf.convert_to_tensor(y[:train_size])
        y_val = tf.convert_to_tensor(y[train_size:train_size+val_size])
        y_test = tf.convert_to_tensor(y[train_size+val_size:])

        return (x_train, y_train), (x_test, y_test), (x_val, y_val)