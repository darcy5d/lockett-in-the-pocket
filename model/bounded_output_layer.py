import tensorflow as tf

class BoundedOutputLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that bounds the output between a minimum and maximum value.
    Uses sigmoid activation and scales the output to the desired range.
    """
    def __init__(self, min_value, max_value, **kwargs):
        super(BoundedOutputLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        
    def build(self, input_shape):
        super(BoundedOutputLayer, self).build(input_shape)
        
    def call(self, inputs):
        scaled_sigmoid = tf.sigmoid(inputs)
        return self.min_value + (self.max_value - self.min_value) * scaled_sigmoid
        
    def get_config(self):
        config = super(BoundedOutputLayer, self).get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        return config
        
    def compute_output_shape(self, input_shape):
        return input_shape 