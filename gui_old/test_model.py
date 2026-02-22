#!/usr/bin/env python3
import os
import sys
import json
import traceback
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Lambda

# Define the custom layer class needed to load the model
class MeanPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=32, **kwargs):
        self.output_dim = output_dim
        super(MeanPoolingLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
        
    def get_config(self):
        config = super(MeanPoolingLayer, self).get_config()
        config.update({"output_dim": self.output_dim})
        return config

def main():
    # Model path
    model_path = os.path.join('model', 'output', 'model.h5')
    
    print(f"Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Debug: Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Create MSE and MAE functions that match the expected API
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
        
    def mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Register the custom layer
    tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer
    
    # Define all custom objects that might be needed for model loading
    custom_objects = {
        'MeanPoolingLayer': MeanPoolingLayer,
        'mean_lambda': lambda x: tf.reduce_mean(x, axis=1),
        # Use custom functions for MSE and MAE
        'mse': mse,
        'mae': mae,
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall(),
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'MSE': mse,
        'MAE': mae,
        'MeanSquaredError': tf.keras.metrics.MeanSquaredError(),
        'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError()
    }
    
    # Try to load the model with custom objects
    try:
        print("Attempting to load model with custom objects...")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully!")
        
        # Print model architecture summary
        print("\nModel Architecture:")
        model.summary()
        
        # Print output layer names and shapes
        print("\nModel Outputs:")
        for i, output in enumerate(model.outputs):
            print(f"  Output {i}: {output.name} (shape: {output.shape})")
            
        # Create dummy input data to test model prediction
        print("\nCreating dummy input data for model prediction...")
        # Determine input shapes from model
        input_shapes = [input.shape for input in model.inputs]
        dummy_inputs = []
        for shape in input_shapes:
            # Create a batch size of 1 for each input
            input_shape = [1] + [dim if dim is not None else 10 for dim in shape[1:]]
            dummy_inputs.append(np.zeros(input_shape))
            
        print(f"Input shapes: {[x.shape for x in dummy_inputs]}")
        
        # Test model prediction
        try:
            print("\nTesting model prediction with dummy data...")
            predictions = model.predict(dummy_inputs)
            
            print("Prediction results:")
            for i, pred in enumerate(predictions):
                print(f"  Output {i}: shape={pred.shape}, data={pred}")
                
            print("\nModel test successful!")
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        
        # Try alternative approach with compile=False
        try:
            print("\nTrying alternative loading approach with compile=False...")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print("Model loaded successfully with compile=False!")
            
            # Print model architecture summary
            print("\nModel Architecture:")
            model.summary()
            
            # Print output layer names and shapes
            print("\nModel Outputs:")
            for i, output in enumerate(model.outputs):
                print(f"  Output {i}: {output.name} (shape: {output.shape})")
        except Exception as e2:
            print(f"Alternative loading approach also failed: {e2}")
            traceback.print_exc()

if __name__ == "__main__":
    main() 