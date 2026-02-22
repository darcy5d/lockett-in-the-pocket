#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import json
import joblib
from pathlib import Path

# Define custom Lambda layer with fixed output shape
def mean_layer(x):
    return tf.reduce_mean(x, axis=1)

# Create a wrapper class for the Lambda layer with explicit output shape
class MeanLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MeanLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# Define custom objects for model loading
custom_objects = {
    'mean_layer': mean_layer,
    'MeanLayer': MeanLayer
}

# Paths to model artifacts
model_dir = Path('../model/output')
model_path = model_dir / 'model.h5'

print(f"Checking if model file exists: {model_path.exists()}")

# Try to load the model
print(f"Attempting to load model from {model_path}...")

try:
    # Try the simplest approach first - loading directly
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully without custom objects!")
    model.summary()
except Exception as e:
    print(f"Direct loading failed: {e}")
    
    try:
        # Try with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully with custom objects!")
        model.summary()
    except Exception as e2:
        print(f"Loading with custom objects failed: {e2}")
        
        try:
            # Try saving model as TF SavedModel format
            print("Attempting alternative loading approach...")
            # Get model's raw config
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e3:
            print(f"Alternative loading approach failed: {e3}")
            
            # Try yet another approach - list model layers via low-level h5py
            try:
                import h5py
                print("\nAttempting to analyze model file with h5py...")
                with h5py.File(model_path, 'r') as f:
                    # List the top-level keys (groups)
                    print("Top-level keys in h5 file:", list(f.keys()))
                    
                    # If 'model_weights' exists, explore its structure
                    if 'model_weights' in f:
                        print("\nLayer names in model_weights:")
                        for layer_name in f['model_weights']:
                            print(f"  - {layer_name}")
                            
                    # Look for model config
                    if 'model_config' in f.attrs:
                        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                        print("\nModel config found!")
                        print("Input layers:", [layer for layer in model_config['config']['input_layers']])
                        print("Output layers:", [layer for layer in model_config['config']['output_layers']])
                        
                        # List all layers
                        print("\nAll layers defined in the model:")
                        for layer in model_config['config']['layers']:
                            print(f"  - {layer['name']} (class: {layer['class_name']})")
                            if layer['class_name'] == 'Lambda':
                                print(f"    Lambda function: {layer.get('config', {}).get('function', [])}")
            except Exception as h5_error:
                print(f"Error analyzing model with h5py: {h5_error}")

print("\nDiagnostic run complete!") 