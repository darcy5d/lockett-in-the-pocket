#!/usr/bin/env python3
"""
AFL Match Predictor GUI Launcher

This script launches the AFL Match Predictor GUI which allows users to:
1. Select home and away teams
2. Choose a venue
3. Select players for both teams from dropdowns
4. Get match predictions based on the trained neural network model

Requirements:
- Python 3.7+
- TensorFlow 2.x
- Tkinter
- NumPy
- Pandas
"""

import os
import sys
import subprocess
import importlib.util

def check_dependency(package_name):
    """Check if a Python package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def main():
    """Main function to check dependencies and run the predictor GUI."""
    # Check for required packages
    required_packages = {
        "tensorflow": "TensorFlow",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "tkinter": "Tkinter"
    }
    
    missing_packages = []
    for package, display_name in required_packages.items():
        if package == "tkinter":
            try:
                import tkinter
            except ImportError:
                missing_packages.append(display_name)
        elif not check_dependency(package):
            missing_packages.append(display_name)
    
    if missing_packages:
        print("Error: Missing required dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        
        print("\nPlease install the missing packages using pip:")
        print("  pip install tensorflow numpy pandas")
        print("\nFor Tkinter (which cannot be installed via pip):")
        print("  - On macOS: brew install python-tk")
        print("  - On Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  - On Windows: Tkinter comes with the standard Python installer")
        
        return 1
    
    # Check if model file exists
    model_path = 'model/output/model.h5'
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("The predictor may not work correctly without the trained model.")
        user_input = input("Do you want to continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return 1
    
    # Run the GUI
    print("Starting AFL Match Predictor GUI...")
    try:
        from gui import predictor_gui
        predictor_gui.create_gui().mainloop()
    except Exception as e:
        print(f"Error starting the GUI: {e}")
        print("\nAlternatively, you can run the GUI directly with:")
        print("  python -m gui.predictor_gui")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 