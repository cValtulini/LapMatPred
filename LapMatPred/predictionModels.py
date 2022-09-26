#########################################################################################
# PACKAGES
#########################################################################################
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import tensorflow.keras.backend as K


#########################################################################################
# CLASSES
#########################################################################################
class LaplacianPredictionModel(tf.keras.Model):

    def __init__(self):
        super().__init__()


class LaplacianPredictionModelFC(LaplacianPredictionModel):

    def __init__(self):
        super().__init__()


    def call(self, inputs):
        pass


class LaplacianPredictionModelCNN(LaplacianPredictionModel):

    def __init__(self):
        super().__init__()


    def call(self, inputs):
        pass


#########################################################################################
# FUNCTIONS
#########################################################################################
def relativeError(y_true, y_pred):
    return tf.norm(
        y_true - y_pred, ord='fro', axis=[-2, -1]
        ) / tf.norm(y_true, ord='fro', axis=[-2, -1])