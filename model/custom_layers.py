import tensorflow as tf

class MeanPoolingLayer(tf.keras.layers.Layer):
    """
    Custom layer to calculate the mean of embeddings across players.
    """
    def __init__(self, output_dim=32, **kwargs):
        self.output_dim = output_dim
        super(MeanPoolingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(MeanPoolingLayer, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config

class BoundedOutputLayer(tf.keras.layers.Layer):
    """
    Custom layer that bounds the output between a minimum and maximum value.
    Uses sigmoid activation and scales the output to the desired range.
    """
    def __init__(self, min_value, max_value, **kwargs):
        super(BoundedOutputLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(BoundedOutputLayer, self).build(input_shape)
        
    def call(self, inputs):
        raw_output = tf.matmul(inputs, self.kernel) + self.bias
        scaled_sigmoid = tf.sigmoid(raw_output)
        return self.min_value + (self.max_value - self.min_value) * scaled_sigmoid
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
        
    def get_config(self):
        config = super(BoundedOutputLayer, self).get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        return config

class ScoreCalculationLayer(tf.keras.layers.Layer):
    """
    Custom layer to calculate team score from goals and behinds.
    Score = goals * 6 + behinds
    """
    def __init__(self, **kwargs):
        super(ScoreCalculationLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        goals, behinds = inputs
        return goals * 6 + behinds
        
    def compute_output_shape(self, input_shape):
        return input_shape[0]
        
    def get_config(self):
        config = super(ScoreCalculationLayer, self).get_config()
        return config

class MarginCalculationLayer(tf.keras.layers.Layer):
    """
    Custom layer to calculate the margin between team scores.
    Margin = team1_score - team2_score
    """
    def __init__(self, **kwargs):
        super(MarginCalculationLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        team1_score, team2_score = inputs
        return team1_score - team2_score
        
    def compute_output_shape(self, input_shape):
        return input_shape[0]
        
    def get_config(self):
        config = super(MarginCalculationLayer, self).get_config()
        return config

class WinProbabilityLayer(tf.keras.layers.Layer):
    """
    Custom layer to convert margin to win probability using sigmoid.
    """
    def __init__(self, scaling_factor=0.1, **kwargs):
        super(WinProbabilityLayer, self).__init__(**kwargs)
        self.scaling_factor = scaling_factor
        
    def call(self, inputs):
        # Scale margin and convert to win probability
        scaled_margin = inputs * self.scaling_factor
        return tf.sigmoid(scaled_margin)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(WinProbabilityLayer, self).get_config()
        config.update({'scaling_factor': self.scaling_factor})
        return config

class DrawProbabilityLayer(tf.keras.layers.Layer):
    """
    Custom layer to calculate draw probability from margin.
    Uses a bell curve centered at 0 margin.
    """
    def __init__(self, concentration=5.0, **kwargs):
        super(DrawProbabilityLayer, self).__init__(**kwargs)
        self.concentration = concentration
        
    def call(self, inputs):
        # Gaussian function centered at 0
        return tf.exp(-tf.square(inputs) * self.concentration)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(DrawProbabilityLayer, self).get_config()
        config.update({'concentration': self.concentration})
        return config

class ProbabilityNormalizationLayer(tf.keras.layers.Layer):
    """
    Custom layer to normalize probabilities so they sum to 1.
    """
    def __init__(self, **kwargs):
        super(ProbabilityNormalizationLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        team1_win, draw, team2_win = inputs
        total = team1_win + draw + team2_win
        return [team1_win / total, draw / total, team2_win / total]
        
    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2]]
        
    def get_config(self):
        config = super(ProbabilityNormalizationLayer, self).get_config()
        return config 