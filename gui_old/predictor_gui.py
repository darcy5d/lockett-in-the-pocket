import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model/output/model.h5')

# Function to make predictions
def make_prediction(team1, team2, location, player_names):
    # Placeholder for data mapping and prediction logic
    # This is where you would map player names to indices and prepare input data
    # For now, we'll just print the inputs
    print(f"Team 1: {team1}, Team 2: {team2}, Location: {location}, Players: {player_names}")
    # Example prediction output
    return "Predicted Outcome"

# Create the main application window
root = tk.Tk()
root.title("AFL Match Predictor")

# Create and place widgets
team1_label = ttk.Label(root, text="Team 1:")
team1_label.grid(column=0, row=0, padx=10, pady=5)
team1_entry = ttk.Entry(root)
team1_entry.grid(column=1, row=0, padx=10, pady=5)

team2_label = ttk.Label(root, text="Team 2:")
team2_label.grid(column=0, row=1, padx=10, pady=5)
team2_entry = ttk.Entry(root)
team2_entry.grid(column=1, row=1, padx=10, pady=5)

location_label = ttk.Label(root, text="Location:")
location_label.grid(column=0, row=2, padx=10, pady=5)
location_entry = ttk.Entry(root)
location_entry.grid(column=1, row=2, padx=10, pady=5)

players_label = ttk.Label(root, text="Player Names (comma-separated):")
players_label.grid(column=0, row=3, padx=10, pady=5)
players_entry = ttk.Entry(root)
players_entry.grid(column=1, row=3, padx=10, pady=5)

# Function to handle prediction button click
def on_predict():
    team1 = team1_entry.get()
    team2 = team2_entry.get()
    location = location_entry.get()
    player_names = players_entry.get().split(",")
    outcome = make_prediction(team1, team2, location, player_names)
    messagebox.showinfo("Prediction", outcome)

# Prediction button
predict_button = ttk.Button(root, text="Predict Outcome", command=on_predict)
predict_button.grid(column=0, row=4, columnspan=2, pady=10)

# Start the application
root.mainloop() 