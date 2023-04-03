#########################################################################################
# PACKAGES
#########################################################################################
from LapMatPred.graphUtilities import *
from LapMatPred.predictionModels import relativeError

import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight

#########################################################################################
# FUNCTIONS
#########################################################################################
def create_dataset(
    rng, nodes_number, connection_probability, number_of_realizations, set_size
):
    """
    Creates a dataset from graphs with the specified number of nodes and connection
    probability following the erdosRenyi model. The dataset is composed by the
    estimated covariance matrix for the graph signal generated following a multivariate
    gaussian distribution with 0-mean vector and covariance equal to the inverse of the
    laplacian for the graph + the identity matrix, which is the ground truth

    Parameters
    ----------
    rng: numpy.random.Generator
        The random number generator for all random operations performed in this function
    nodes_number: int
        The number of nodes for the graphs
    connection_probability: float
        The connection probability for the graphs
    number_of_realizations: int
        The number of signals to generate for each estimation of the covariance matrix
    set_size: int
        The number of samples to generate

    Returns
    -------
    numpy.ndarray: shape=(set_size, nodes_number, nodes_number)
        The estimated covariance matrices
    numpy.ndarray: shape=(set_size, nodes_number, nodes_number)
        The ground truths generalized laplacian matrices

    """
    adjacency = np.array(
        [
            erdos_renyi(nodes_number, connection_probability, rng=rng)
            for _ in range(set_size)
        ]
    )
    laplacian = np.array([graph_laplacian(a) for a in adjacency])

    q_true = np.eye(nodes_number) + laplacian

    k_matrix = np.linalg.inv(q_true)

    mean_vector = np.zeros(nodes_number)
    covariance = np.array(
        [
            np.cov(
                rng.multivariate_normal(
                    mean_vector, k_matrix[i], size=number_of_realizations
                ),
                rowvar=False,
            )
            for i in range(k_matrix.shape[0])
        ]
    )
    return covariance, q_true


def split_dataset(x, y, train_size, val_size=None):
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
    val_size: int, default None
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
        print("Error: x and y have different number of samples")
        return None
    if val_size is None:
        if x.shape[0] < train_size:
            print("Error: train_size is greater than the number of samples")
            return None

        x_train = tf.convert_to_tensor(x[:train_size])
        x_test = tf.convert_to_tensor(x[train_size:])
        y_train = tf.convert_to_tensor(y[:train_size])
        y_test = tf.convert_to_tensor(y[train_size:])

        return (x_train, y_train), (x_test, y_test)
    else:
        if x.shape[0] < train_size or x.shape[0] < train_size + val_size:
            print(
                "Error: dataset size is not equal to the sum of train and test set sizes"
            )
            return None

        x_train = tf.convert_to_tensor(x[:train_size])
        x_val = tf.convert_to_tensor(x[train_size : train_size + val_size])
        x_test = tf.convert_to_tensor(x[train_size + val_size :])

        y_train = tf.convert_to_tensor(y[:train_size])
        y_val = tf.convert_to_tensor(y[train_size : train_size + val_size])
        y_test = tf.convert_to_tensor(y[train_size + val_size :])

        return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def quantize_for_classification(x, number_of_classes=4, low=0, high=1, avoid_zero=True):
    """
    Quantizes the input data into a specified number of classes. If avoid_zero is True,
    moves all nonzero elements in the input data to the next class.

    Parameters
    ----------
    x: numpy.ndarray
        The input data
    number_of_classes: int, default 4
        The number of classes to quantize the input data into
    low: float, default 0
        The lower bound of the quantization
    high: float, default 1
        The upper bound of the quantization
    avoid_zero: bool, default True
        If True, moves all nonzero elements in the input data to the next class

    Returns
    -------
    numpy.ndarray:
        The class' index for each element in the input data
    numpy.ndarray:
        The quantized data

    """
    levels = np.linspace(low, high, number_of_classes + 1)[:-1]
    x_classes = np.argmin(np.absolute(x[..., np.newaxis] - levels), axis=-1)

    if avoid_zero:
        # We don't want elements different from 0 to be classified as zeros
        x_classes = np.where((x_classes == 0) == (x == 0), x_classes, x_classes + 1)

    return x_classes, levels[x_classes]


def create_sample_weight_mask(y, classification=True):
    """
    Creates a mask for the sample weights. If classification is True, the mask is giveb
    by class weights for each sample, otherwise the function is intended to be used to
    discriminate between zero and nonzero elements in the input data for sparse arrays,
    computing weights for the two cases as if they were classes.

    Parameters
    ----------
    y: numpy.ndarray or tensorflow.Tensor
        The input data used to extract classes, compute weights and create the mask
    classification: bool, default True
        If True, classes are extracted from values of y, otherwise the class of an element
        is represented by it being zero or nonzero

    Returns
    -------
    numpy.ndarray: y.shape
        The mask of sample weights
    dict: {int: float}
        A dictionary containing couples of {class, class's weight}}

    """

    if classification:
        classes = np.unique(y)
        # Function wants a flattened array as y
        weights = class_weight.compute_class_weight(
            "balanced", classes=classes, y=y.numpy().flatten()
        )
    else:
        classes = np.array([0, 1])
        y = np.where(y > 0, 1, 0)
        weights = class_weight.compute_class_weight(
            "balanced", classes=classes, y=y.flatten()
        )
        y = y.reshape(y.shape[0], y.shape[1], 1)

    return weights[y], {key: value for (key, value) in zip(classes, weights)}


def estimate_error(n_classes, nodes_number, connection_probability, size=1000):
    """
    Estimates the error of a perfect classification of a sample based on the
    number of classes used for the quantization.

    Parameters
    ----------
    n_classes: int | Iterable(int)
        The number of classes used for the quantization
    nodes_number: int
        The number of nodes in the graph
    connection_probability: float
        The probability of a connection between two nodes
    size: int, default 1000
        The number of samples to create for the estimation

    Returns
    -------
    Dict(int, float):
        A dictionary containing key, value pair, with the number of classes as
        key and the estimated error as value

    """
    errors = []

    _, dataset = create_dataset(
        # Seed shouldn't be set here but anyway
        np.random.default_rng(seed=42),
        nodes_number,
        connection_probability,
        2,  # We only care about Q_true
        size,
    )
    dataset = np.absolute(
        [dataset[i] - np.diag(np.diag(dataset[i])) for i in range(size)]
    )
    tf_dataset = tf.convert_to_tensor(dataset)

    for number_classes in list(n_classes):
        # We don't care about the classes, only about the quantized value of
        # weights
        _, Q_true = quantize_for_classification(dataset, number_classes)
        tf_Q_true = tf.convert_to_tensor(Q_true)

        error = relative_error(tf_dataset, tf_Q_true)
        errors.append(tf.reduce_sum(error).numpy() / size)

    return {key: value for (key, value) in zip(n_classes, errors)}
