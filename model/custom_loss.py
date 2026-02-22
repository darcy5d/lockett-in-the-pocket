import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package='Custom')
def mse(y_true, y_pred):
    """
    Mean Squared Error loss function that is properly serializable.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

@register_keras_serializable(package='Custom')
def mae(y_true, y_pred):
    """
    Mean Absolute Error loss function that is properly serializable.
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Define other custom loss functions here as needed 