################################################################################
# PACKAGES
################################################################################
import tensorflow as tf
from tensorflow.keras import layers, models, losses


################################################################################
# CLASSES
################################################################################
# MODELS
################################################################################
class LaplacianPredictionModel(tf.keras.Model):
    """
    General class to be inherited by all the models, mainly containing common parameters
    and methods
    """

    def __init__(self, nodes_number):
        super().__init__()
        self.nodes_number = nodes_number

    def get_config(self):
        config = super().get_config()
        config.update({"nodes_number": self.nodes_number})
        return config


class LaplacianPredictionModelFC(LaplacianPredictionModel):
    """
    Model mainly implementing fully connected layers, after each FFN 1x1 convolution is
    applied concatenating the current output with the previous' layer one to preserve
    information about previous representations and to reduce backpropagation problems.
    """

    def __init__(self, nodes_number, depth=1, activation="relu", h_activation="relu"):
        super().__init__(nodes_number)

        nn = self.nodes_number

        self.flatten = layers.Reshape((nn**2,), input_shape=(nn, nn))
        self.normalize = [layers.BatchNormalization(axis=-1) for _ in range(depth)]

        self.ffn = [
            layers.Dense(nn**2, activation=h_activation) for _ in range(depth)
        ]
        self.conv = [
            layers.Conv2D(1, 1, activation=activation, input_shape=(nn, nn, 2))
            for _ in range(depth)
        ]

        self.output_layer = layers.Dense(nn * (nn - 1) // 2, activation=activation)

        self.drop = layers.Dropout(0.2)
        self.reshape = layers.Reshape((nn, nn, 1), input_shape=(nn**2,))
        self.concat = layers.Concatenate(axis=-1)

        del nn

    def call(self, inputs):
        # Flattens the input to be fed to the FFN
        x = self.flatten(inputs)

        for ffn_layer, conv_layer, norm_layer in zip(
            self.ffn, self.conv, self.normalize
        ):
            # Creates new copy of current input to be given to 1x1 Convolutional Layer
            x = norm_layer(x)
            previous = self.reshape(x)

            x = self.reshape(self.drop(ffn_layer(x)))
            x = self.flatten(conv_layer(self.concat([x, previous])))

        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({"flatten": self.flatten})
        config.update({"normalize": self.normalize})
        config.update({"ffn": self.ffn})
        config.update({"conv": self.conv})
        config.update({"output_layer": self.output_layer})
        config.update({"drop": self.drop})
        config.update({"reshape": self.reshape})
        config.update({"concat": self.concat})
        return config


class LaplacianPredictionModelFCSimple(LaplacianPredictionModel):
    # TODO: Rewrite description
    """
    Model mainly implementing fully connected layers, after each FFN 1x1 convolution is
    applied concatenating the current output with the previous' layer one to preserve
    information about previous representations and to reduce backpropagation problems.
    """

    def __init__(self, nodes_number, depth=1, activation="relu", h_activation="relu"):
        super().__init__(nodes_number)

        nn = self.nodes_number

        self.flatten = layers.Reshape((nn**2,), input_shape=(nn, nn))
        self.normalize = layers.BatchNormalization(axis=-1)

        self.ffn = [
            layers.Dense(nn**2, activation=h_activation) for _ in range(depth)
        ]
        self.ffn_2 = layers.Dense(nn**2 // 2, activation=h_activation)

        self.output_layer = layers.Dense(nn * (nn - 1) // 2, activation=activation)

        self.drop = layers.Dropout(0.2)

        del nn

    def call(self, inputs):
        # Flattens the input to be fed to the FFN
        x = self.flatten(inputs)
        x = self.normalize(x)

        for ffn_layer in self.ffn:
            x = self.drop(ffn_layer(x))
        x = self.drop(self.ffn_2(x))

        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({"flatten": self.flatten})
        config.update({"normalize": self.normalize})
        config.update({"ffn": self.ffn})
        config.update({"ffn": self.ffn_2})
        config.update({"output_layer": self.output_layer})
        config.update({"drop": self.drop})
        return config


class LaplacianPredictionModelFCParallel(LaplacianPredictionModel):
    # TODO: Rewrite description
    """
    Model mainly implementing fully connected layers, after each FFN 1x1 convolution is
    applied concatenating the current output with the previous' layer one to preserve
    information about previous representations and to reduce backpropagation problems.
    """

    def __init__(
        self, nodes_number, depth=1, width=2, activation="relu", h_activation="relu"
    ):
        super().__init__(nodes_number)

        nn = self.nodes_number

        self.flatten = layers.Reshape((nn**2,), input_shape=(nn, nn))
        self.normalize = layers.BatchNormalization(axis=-1)

        self.ffn = [
            [layers.Dense(nn**2, activation=h_activation) for _d in range(depth)]
            for _w in range(width)
        ]
        self.conv = layers.Conv2D(
            1, 1, activation=activation, input_shape=(nn, nn, width)
        )

        self.output_layer = layers.Dense(nn * (nn - 1) // 2, activation=activation)

        self.drop = layers.Dropout(0.2)
        self.reshape = layers.Reshape((nn, nn, 1), input_shape=(nn**2,))
        self.concat = layers.Concatenate(axis=-1)

        del nn

    def call(self, inputs):
        # Flattens the input to be fed to the FFN
        x = self.flatten(inputs)
        x = self.normalize(x)
        x_array = [self.drop(ffn_stack[0](x)) for ffn_stack in self.ffn]

        if len(self.ffn[0]) > 1:
            for w, ffn_stack in enumerate(self.ffn):
                for ffn_layer in ffn_stack[1:]:
                    x_array[w] = self.drop(ffn_layer(x_array[w]))

                x_array[w] = self.reshape(x_array(w))
        x = self.flatten(self.conv(self.concat(x_array)))

        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({"flatten": self.flatten})
        config.update({"normalize": self.normalize})
        config.update({"ffn": self.ffn})
        config.update({"conv": self.conv})
        config.update({"output_layer": self.output_layer})
        config.update({"drop": self.drop})
        config.update({"reshape": self.reshape})
        config.update({"concat": self.concat})
        return config


################################################################################
# FUNCTIONS
################################################################################
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
    return tf.norm(y_true - y_pred, axis=-1) / tf.norm(y_true, axis=-1)


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
        dtype=tf.float32,
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
        dtype=tf.float32,
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
        tf.math.logical_and(
            tf.math.logical_not(edges), tf.math.logical_not(pred_edges)
        ),
        dtype=tf.float32,
    )
    false_p = tf.cast(
        tf.math.logical_and(tf.math.logical_not(edges), pred_edges),
        dtype=tf.float32,
    )
    false_n = tf.cast(
        tf.math.logical_and(edges, tf.math.logical_not(pred_edges)),
        dtype=tf.float32,
    )

    n_true_p = tf.reduce_sum(true_p)
    n_true_n = tf.reduce_sum(true_n)
    n_false_p = tf.reduce_sum(false_p)
    n_false_n = tf.reduce_sum(false_n)

    return (n_true_p + n_true_n) / (n_true_p + n_true_n + n_false_p + n_false_n)
