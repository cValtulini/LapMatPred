################################################################################
# PACKAGES
################################################################################
import os

from LapMatPred.graphUtilities import *
from LapMatPred.predictionModels import relativeError

import numpy as np
import tensorflow as tf


################################################################################
# CONSTANTS
################################################################################
Q_FEATURES_DESCRIPTION = {
    "q_raw": tf.io.FixedLenFeature([], tf.string),
}

K_FEATURES_DESCRIPTION = {
    "k_raw": tf.io.FixedLenFeature([], tf.string),
}


################################################################################
# FUNCTIONS
################################################################################
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_covariance_sample(rng, q_true, number_of_realizations):
    """
    Creates a covariance sample starting from a GLP matrix q_true, with number_of_realizations realizations of a gaussian signal defined on the GLP used to estimate the covariance.

    Parameters
    ----------
    rng: numpy.random.Generator
        The numpy random number generator for all random operations performed in the function.
    q_true: tensorflow.Tensor
        The GLP Tensor, with ndim equal to 2 (matrix), from which the covariance is obtained.
    number_of_realizations: int
        The number of signals to generate for each estimation of the covariance matrix.

    Returns
    -------
    numpy.ndarray: shape=(nodes_number, nodes_number)
        The estimated covariance matrix
    """
    k_matrix = np.linalg.inv(q_true)

    covariance = np.cov(
        rng.multivariate_normal(
            np.zeros(k_matrix.shape[0]), k_matrix, size=number_of_realizations
        ),
        rowvar=False,
    )
    return covariance


def create_q_true_tfr_set(rng, nodes_number, connection_probability, set_size, path):
    """
    Creates a dataset of GLP matrices in the tfrecords format.

    Args:
        rng (numpy.random.Generator): The numpy random number generator for all random operations performed in the function.
        nodes_number (int): The number of nodes of each GLP matrix.
        connection_probability (float): The probability of an edge between two nodes of the graph.
        set_size (int): The total number of GLP samples in the dataset.
        path (str): Where to save the dataset, includes path and filename. File extension is added automatically.
    """
    filename = os.path.join(path + ".tfrecords")
    print("Writing", filename)

    writer = tf.io.TFRecordWriter(filename)
    for index in range(set_size):
        laplacian = graph_laplacian(
            erdos_renyi(nodes_number, connection_probability, rng=rng)
        )

        q = np.eye(nodes_number) + laplacian

        q_raw = tf.io.serialize_tensor(tf.convert_to_tensor(q))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "q_raw": _bytes_feature(q_raw.numpy()),
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


def create_covariance_tfr_set(
    rng,
    q_true_set,
    number_of_realizations,
    save_path,
):
    """
    Creates a dataset of covariance matrices in the tfrecords format starting from a set of GLP matrices.

    Args:
        rng (numpy.random.Generator): The numpy random number generator for all random operations performed in the function.
        q_true_set (_type_): The set of GLP matrices from which the covariance set is estimated.
        number_of_realizations (int): The number of realizations used to generate gaussian signals from which each covariance matrix is obtained.
        save_path (str): Where to save the dataset, includes path and filename. File extension is added automatically.
    """

    filename = os.path.join(save_path + ".tfrecords")
    print("Writing", filename)

    writer = tf.io.TFRecordWriter(filename)

    for q in q_true_set:
        k = create_covariance_sample(rng, q, number_of_realizations)

        k_raw = tf.io.serialize_tensor(tf.convert_to_tensor(k))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "k_raw": _bytes_feature(k_raw.numpy()),
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


def _parse_true_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.

    parsed_record = tf.io.parse_single_example(example_proto, Q_FEATURES_DESCRIPTION)

    q = tf.io.parse_tensor(parsed_record["q_raw"], out_type=tf.float64)

    return q


def _parse_k_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.

    parsed_record = tf.io.parse_single_example(example_proto, K_FEATURES_DESCRIPTION)

    k = tf.io.parse_tensor(parsed_record["k_raw"], out_type=tf.float64)

    return k


def _map_triu_function(q):
    """
    Used to map element of a tf.data.Dataset to an array made of their element contained in the upper triangular section of the matrix (main diagonal excluded).

    Args:
        q (tensorflow.Tensor): The Tensor element from which the upper triangular is extracted.

    Returns:
        tensorflow.Tensor: A 1D Tensor made of the upper triangular elements, main diagonal excluded, of the input Tensor. The upper triangular is flattened "row-wise" e.g. [[1, 2, 3], [2, 4, 5], [3, 6, 7]] becomes [2, 3, 5].
    """
    ones = tf.ones_like(q)
    mask_a = tf.linalg.band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
    mask.set_shape([None, None])

    q_triu_flat = tf.boolean_mask(q, mask)

    return tf.math.abs(q_triu_flat)


def _reconstruct_triu_estimate_function(estimate, nodes_number):
    """
    Reconstructs a matrix from a 1D Tensor of its upper triangular elements, main diagonal excluded. Constructs a symmetric matrix repositioning the input array's elements and then builds the main diagonal elements by summing the elements of the corresponding rows.

    Args:
        estimate (tensorflow.Tensor): A 1D Tensor of the upper triangular elements of the matrix to be reconstructed.
        nodes_number (int): The number of nodes of the graph corresponding to the reconstructed matrix i.e., the size of the two dimensions of the matrix.

    Returns:
        tensorflow.Tensor: The reconstructed matrix.
    """
    full = tf.zeros(shape=(nodes_number, nodes_number), dtype=estimate.dtype)
    upper_triangular = tf.linalg.band_part(
        tf.ones_like(full), 0, -1
    ) - tf.linalg.band_part(tf.ones_like(full), 0, 0)
    indices = tf.where(tf.equal(upper_triangular, 1))

    full = tf.tensor_scatter_nd_update(full, indices, -estimate)
    full += tf.transpose(full)

    full -= tf.linalg.diag(tf.reduce_sum(full, axis=-1))

    return full
