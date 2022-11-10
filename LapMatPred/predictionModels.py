#########################################################################################
# PACKAGES
#########################################################################################
import tensorflow as tf
from tensorflow.keras import layers, models, losses


# import tensorflow.keras.backend as K


#########################################################################################
# CLASSES
#########################################################################################
# MODELS
#########################################################################################
class LaplacianPredictionModel(tf.keras.Model):
    """
    General class to be inherited by all the models, mainly containing common parameters
    and methods
    """


    def __init__(self, nodes_number):
        super().__init__()
        self.nodes_number = nodes_number


    def loadSavedModel(self, path):
        pass


class LaplacianPredictionModelFC(LaplacianPredictionModel):
    """
    Model mainly implementing fully connected layers, after each FFN 1x1 convolution is
    applied concatenating the current output with the previous' layer one to preserve
    information about previous representations and to reduce backpropagation problems.
    """


    def __init__(self, nodes_number, depth=1, activation='relu'):
        super().__init__(nodes_number)

        self.flatten = layers.Reshape(
            (nodes_number ** 2,), input_shape=(nodes_number, nodes_number)
            )
        self.ffn = [
                layers.Dense(nodes_number ** 2, activation=activation) for _ in
                range(depth)
                ]
        self.conv = [
                layers.Conv2D(
                    1, 1,
                    activation=activation, input_shape=(nodes_number, nodes_number, 2)
                    ) for _ in range(depth)
                ]
        self.output_layer = layers.Dense(
            nodes_number * (nodes_number - 1) // 2, activation=activation
            )
        self.drop = layers.Dropout(0.2)
        self.reshape = layers.Reshape(
            (nodes_number, nodes_number, 1), input_shape=(nodes_number ** 2,)
            )
        self.concat = layers.Concatenate(axis=-1)


    def call(self, inputs):
        # Flattens the input to be fed to the FFN
        x = self.flatten(inputs)

        for ffn_layer, conv_layer in zip(self.ffn, self.conv):
            # Creates new copy of current input to be given to 1x1 Convolutional Layer
            previous = self.reshape(x)

            x = self.reshape(self.drop(ffn_layer(x)))
            x = self.flatten(conv_layer(self.concat([x, previous])))

        return self.output_layer(x)


class LaplacianPredictionModelQuantizedClassification(LaplacianPredictionModel):

    # The idea is to quantize the Laplacian matrix in order to have a classification
    # problem. Where the classes are given by the quantization levels an element of the
    # matrix is assigned to.

    def __init__(self, nodes_number, classes):
        super().__init__(nodes_number)
        self.classes = classes

        self.flatten = layers.Reshape(
            (nodes_number ** 2,), input_shape=(nodes_number, nodes_number)
            )
        self.ffn = [
                layers.Dense(
                    nodes_number * (nodes_number - 1) // 2, activation='relu'
                    )
                for _ in range(self.classes * 2)
                ]
        self.reshape = layers.Reshape((nodes_number*(nodes_number-1)//2, 1))
        self.concat = layers.Concatenate(axis=-1)

        self.output_layer = layers.Conv1D(self.classes, 1)


    def call(self, inputs):
        inputs = self.flatten(inputs)

        x = [ffn_layer(inputs) for ffn_layer in self.ffn]

        x = [self.reshape(element) for element in x]

        return self.output_layer(self.concat(x))


# LAYERS
#########################################################################################


#########################################################################################
# FUNCTIONS
#########################################################################################
def relativeError(y_true, y_pred):
    """
    Computes the relative error between true and predicted values.
    The error is given by || y_true - y_pred || / || y_true || where || . || is the
    Frobenius norm.

    Parameters
    ----------
    y_true: tensorflow.Tensor
        True values, shape (batch_size, N, N)
    y_pred: tensorflow.Tensor
        Predicted values, shape (batch_size, N, N)

    Returns
    -------
    tensorflow.Tensor
        The relative error, as defined

    """
    return tf.norm(
        y_true - y_pred, ord='fro', axis=[-2, -1]
        ) / tf.norm(y_true, ord='fro', axis=[-2, -1])


def edgesPrecision(y_true, y_pred):
    """
    Computes the precision in identifying edges of a model, meaning that weights' values
    are not considered. The precision is given by TP / (TP + FP) where TP is the number
    of true positives, i.e. the number of edges correctly identified, and FP is the number
    of false positives, i.e. the number of edges wrongly identified.

    Parameters
    ----------
    y_true: tensorflow.Tensor
        True values, shape (batch_size, N, N)
    y_pred: tensorflow.Tensor
        Predicted values, shape (batch_size, N, N)

    Returns
    -------
    tensorflow.Tensor
        The precision, as defined

    """

    edges = tf.math.logical_not(tf.experimental.numpy.isclose(y_true, 0))
    pred_edges = tf.math.logical_not(tf.experimental.numpy.isclose(y_pred, 0))

    true_p = tf.cast(tf.math.logical_and(edges, pred_edges), dtype=tf.float32)
    false_p = tf.cast(
        tf.math.logical_and(tf.math.logical_not(edges), pred_edges),
        dtype=tf.float32
        )

    return tf.reduce_sum(true_p) / (tf.reduce_sum(true_p) + tf.reduce_sum(false_p))


def edgesRecall(y_true, y_pred):
    """
    Computes the recall in identifying edges of a model, meaning that weights' values
    are not considered. The recall is given by TP / (TP + FN) where TP is the number
    of true positives, i.e. the number of edges correctly identified, and FN is the number
    of false negatives, i.e. the number of edges wrongly not identified.

    Parameters
    ----------
    y_true: tensorflow.Tensor
        True values, shape (batch_size, N, N)
    y_pred: tensorflow.Tensor
        Predicted values, shape (batch_size, N, N)

    Returns
    -------
    tensorflow.Tensor
        The recall, as defined

    """

    edges = tf.math.logical_not(tf.experimental.numpy.isclose(y_true, 0))
    pred_edges = tf.math.logical_not(tf.experimental.numpy.isclose(y_pred, 0))

    true_p = tf.cast(tf.math.logical_and(edges, pred_edges), dtype=tf.float32)
    false_n = tf.cast(
        tf.math.logical_and(edges, tf.math.logical_not(pred_edges)),
        dtype=tf.float32
        )

    return tf.reduce_sum(true_p) / (tf.reduce_sum(true_p) + tf.reduce_sum(false_n))


def edgesAccuracy(y_true, y_pred):
    """
    Computes the accuracy in identifying edges of a model, meaning that weights' values
    are not considered. The accuracy is given by (TP + TN) / (TP + TN + FP + FN) where TP
    is the number of true positives, i.e. the number of edges correctly identified, TN is
    the number of true negatives, i.e. the number of non-edges correctly identified, FP is
    the number of false positives, i.e. the number of edges wrongly identified, and FN is
    the number of false negatives, i.e. the number of edges wrongly not identified.

    Parameters
    ----------
    y_true: tensorflow.Tensor
        True values, shape (batch_size, N, N)
    y_pred: tensorflow.Tensor
        Predicted values, shape (batch_size, N, N)

    Returns
    -------
    tensorflow.Tensor
        The accuracy, as defined

    """

    edges = tf.math.logical_not(tf.experimental.numpy.isclose(y_true, 0))
    pred_edges = tf.math.logical_not(tf.experimental.numpy.isclose(y_pred, 0))

    true_p = tf.cast(tf.math.logical_and(edges, pred_edges), dtype=tf.float32)
    true_n = tf.cast(
        tf.math.logical_and(tf.math.logical_not(edges), tf.math.logical_not(pred_edges)),
        dtype=tf.float32
        )
    false_p = tf.cast(
        tf.math.logical_and(tf.math.logical_not(edges), pred_edges),
        dtype=tf.float32
        )
    false_n = tf.cast(
        tf.math.logical_and(edges, tf.math.logical_not(pred_edges)),
        dtype=tf.float32
        )

    n_true_p = tf.reduce_sum(true_p)
    n_true_n = tf.reduce_sum(true_n)
    n_false_p = tf.reduce_sum(false_p)
    n_false_n = tf.reduce_sum(false_n)

    return (n_true_p + n_true_n) / (n_true_p + n_true_n + n_false_p + n_false_n)
