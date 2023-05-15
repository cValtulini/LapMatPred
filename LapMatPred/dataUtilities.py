################################################################################
# PACKAGES
################################################################################
import os

from LapMatPred.graphUtilities import *
from LapMatPred.predictionModels import relativeError

import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight


################################################################################
# CONSTANTS
################################################################################
FEATURES_DESCRIPTION = feature_description = {
    "covariance_raw": tf.io.FixedLenFeature([], tf.string),
    "q_true_raw": tf.io.FixedLenFeature([], tf.string),
}

################################################################################
# FUNCTIONS
################################################################################
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_sample(rng, nodes_number, connection_probability, number_of_realizations):
    # TODO: Modify doc (copy pasted from create_dataset)
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
    adjacency = erdos_renyi(nodes_number, connection_probability, rng=rng)
    laplacian = graph_laplacian(adjacency)

    q_true = np.eye(nodes_number) + laplacian

    k_matrix = np.linalg.inv(q_true)

    mean_vector = np.zeros(nodes_number)
    covariance = np.cov(
        rng.multivariate_normal(mean_vector, k_matrix, size=number_of_realizations),
        rowvar=False,
    )
    return covariance, q_true


def create_tfr_set(
    rng, nodes_number, connection_probability, number_of_realizations, set_size, path
):
    """Converts a dataset to tfrecords."""

    filename = os.path.join(path + ".tfrecords")
    print("Writing", filename)

    writer = tf.io.TFRecordWriter(filename)
    for index in range(set_size):
        sample = create_sample(
            rng, nodes_number, connection_probability, number_of_realizations
        )

        covariance_tf = tf.convert_to_tensor(sample[0])
        q_true_tf = tf.convert_to_tensor(sample[1])

        covariance_raw = tf.io.serialize_tensor(covariance_tf)
        q_true_raw = tf.io.serialize_tensor(q_true_tf)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "covariance_raw": _bytes_feature(covariance_raw.numpy()),
                    "q_true_raw": _bytes_feature(q_true_raw.numpy()),
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.

    parsed_record = tf.io.parse_single_example(example_proto, FEATURES_DESCRIPTION)

    covariance = tf.io.parse_tensor(
        parsed_record["covariance_raw"], out_type=tf.float64
    )
    q_true = tf.io.parse_tensor(parsed_record["q_true_raw"], out_type=tf.float64)

    return covariance, q_true


def _map_triu_function(covariance, q_true):
    ones = tf.ones_like(q_true)
    mask_a = tf.linalg.band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
    mask.set_shape([None, None])

    q_true_flat = tf.boolean_mask(q_true, mask)

    return covariance, tf.math.abs(q_true_flat)
