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
    # TODO: write
    """
    Doc

    Parameters
    ----------
    rng:
    q_true:
    number_of_realizations: int
        The number of signals to generate for each estimation of the covariance matrix

    Returns
    -------
    numpy.ndarray: shape=(set_size, nodes_number, nodes_number)
        The estimated covariance matrices

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
    """Creates a dataset in the tfrecords format."""

    filename = os.path.join(path + ".tfrecords")
    print("Writing", filename)

    writer = tf.io.TFRecordWriter(filename)
    for index in range(set_size):
        laplacian = graph_laplacian(
            erdos_renyi(nodes_number, connection_probability, rng=rng)
        )

        q = np.eye(nodes_number) + laplacian

        q_tf = tf.convert_to_tensor(q)
        q_raw = tf.io.serialize_tensor(q_tf)

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
    """Creates a dataset in the tfrecords format."""

    filename = os.path.join(save_path + ".tfrecords")
    print("Writing", filename)

    writer = tf.io.TFRecordWriter(filename)

    for q in q_true_set:
        k = create_covariance_sample(rng, q, number_of_realizations)

        k_tf = tf.convert_to_tensor(k)
        k_raw = tf.io.serialize_tensor(k_tf)

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
    ones = tf.ones_like(q)
    mask_a = tf.linalg.band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
    mask.set_shape([None, None])

    q_triu_flat = tf.boolean_mask(q, mask)

    return tf.math.abs(q_triu_flat)
