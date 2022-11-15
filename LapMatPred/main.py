#########################################################################################
# PACKAGES
#########################################################################################
from LapMatPred.graphUtilities import *

import numpy as np

import tensorflow as tf
from keras import layers, models

#########################################################################################
# MAIN CODE
#########################################################################################
if __name__ == "__main__":
    rng = np.random.RandomState(42)

    train_set_size = 800
    test_set_size = 200
    set_size = train_set_size + test_set_size

    nodes_number = 20  # number of nodes in the graph
    connection_prob = 0.3  # probability of connection between two nodes

    print("Generating data...")

    adjacency = np.array(
        [erdosRenyi(nodes_number, connection_prob) for _ in range(set_size)]
    )
    laplacian = np.array([graphLaplacian(a) for a in adjacency])

    Q_true = np.eye(nodes_number) + laplacian

    K_matrix = np.linalg.inv(Q_true)

    x_train = tf.convert_to_tensor(K_matrix[:train_set_size])
    x_test = tf.convert_to_tensor(K_matrix[train_set_size:])

    y_train = tf.convert_to_tensor(Q_true[:train_set_size])
    y_test = tf.convert_to_tensor(Q_true[train_set_size:])

    print("Data generated!")
    print("Building model...")

    model = models.Sequential(
        [
            layers.Flatten(input_shape=(nodes_number, nodes_number)),
            layers.Dense(nodes_number**2, activation="relu"),
            layers.Dense(nodes_number**2, activation="relu"),
            layers.Reshape((nodes_number, nodes_number)),
        ]
    )

    print("Model built!")
    print("Compiling model...")

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    print("Model compiled!")
    print("Training model...")

    model.fit(x_train, y_train, epochs=1)

    print("Model trained!")
    print("Evaluating model...")

    model.evaluate(x_test, y_test)

    print("Model evaluated!")
