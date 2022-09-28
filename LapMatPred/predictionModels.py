#########################################################################################
# PACKAGES
#########################################################################################
import tensorflow as tf
from tensorflow.keras import layers, models, losses
# import tensorflow.keras.backend as K


#########################################################################################
# CLASSES
#########################################################################################
class LaplacianPredictionModel(tf.keras.Model):

    def __init__(self, nodes_number):
        super().__init__()
        self.nodes_number = nodes_number


class LaplacianPredictionModelFC(LaplacianPredictionModel):

    def __init__(self, nodes_number, depth, depth_layers, activation='selu'):
        super().__init__(nodes_number)

        self.flatten = layers.Flatten(input_shape=(nodes_number, nodes_number))
        for i in range(depth-1):
            self.condense_layers = [
                    [
                        layers.Dense(
                            nodes_number//(i+1), activation=activation
                            ) for _ in range(depth_layers)

                        ] for i in range(depth)
                    ]
        self.middle_layers = [
                layers.Dense(
                    nodes_number//depth, activation=activation
                    ) for _ in range(depth_layers)
                ]
        for i in range(depth-1):
            self.regrow_layers = [
                    [
                        layers.Dense(
                            nodes_number//(i+1), activation=activation
                            ) for _ in range(depth_layers)
                        ] for i in range(depth)
                    ]

        self.drops = [layers.Dropout(0.2) for _ in range(3)]

        self.original_dim_layer = layers.Dense(nodes_number**2, activation=activation)
        self.output_layer = layers.Reshape((nodes_number, nodes_number))

    def call(self, inputs):
        x = self.flatten(inputs)

        for stack in self.condense_layers:
            for element in stack:
                x = element(x)

        x = self.drops[0](x)

        for element in self.middle_layers:
            x = element(x)

        x = self.drops[1](x)

        for stack in self.regrow_layers[::-1]:
            for lay in stack:
                x = lay(x)

        x = self.drops[2](x)

        x = self.original_dim_layer(x)
        return self.output_layer(x)


class LaplacianPredictionModelCNN(LaplacianPredictionModel):

    def __init__(self, nodes_number):
        super().__init__(nodes_number)

    def call(self, inputs):
        pass


#########################################################################################
# FUNCTIONS
#########################################################################################
def relativeError(y_true, y_pred):
    return tf.norm(
        y_true - y_pred, ord='fro', axis=[-2, -1]
        ) / tf.norm(y_true, ord='fro', axis=[-2, -1])
