# AFL Match Predictor

This directory contains tools for predicting AFL match outcomes based on team selections, venues, and player lineups.

## Prerequisites

- Python 3.9 or higher
- TensorFlow (for prediction functionality)
- Tkinter (for GUI version) or Streamlit (for web-based version)

## Available Tools

### Web-based Version (predictor_streamlit.py) - RECOMMENDED

A modern web-based interface built with Streamlit for making match predictions. This version is recommended as it supports data visualization and is easier to set up than the Tkinter version.

**Requirements:**
- Streamlit (`pip install streamlit`)
- TensorFlow (optional, but required for predictions)

**Usage:**
```
pip install streamlit
streamlit run gui/predictor_streamlit.py
```

This will open a web browser with the application interface.

**Features:**
- Tabbed interface for easy navigation
- Multi-select player selection
- Visual prediction results
- Ready for future player statistics visualizations
- Works in any modern web browser

### GUI Version (predictor_gui.py)

A graphical user interface for making match predictions with dropdown menus for team, venue, and player selections.

**Requirements:**
- Tkinter must be installed
- TensorFlow (optional, but required for predictions)

**Usage:**
```
python gui/predictor_gui.py
```

If Tkinter is not available, the script will provide instructions for installing it.

### Command-Line Version (predictor_cli.py)

A text-based interface for making match predictions when a graphical interface is not available or preferred.

**Requirements:**
- TensorFlow (optional, but required for predictions)

**Usage:**
```
python gui/predictor_cli.py
```

## Getting Started

1. Make sure you're in the virtual environment:
   ```
   source afl_venv_3_9/bin/activate
   ```

2. Install the required dependencies:
   ```
   pip install tensorflow streamlit
   ```

3. Run your preferred version:
   ```
   # For web-based version (recommended)
   streamlit run gui/predictor_streamlit.py
   
   # For GUI version
   python gui/predictor_gui.py
   
   # For command-line version
   python gui/predictor_cli.py
   ```

## Prediction Process

The prediction tool follows these steps:
1. Select home and away teams
2. Select the venue for the match
3. Select players for both teams (at least 18 per team)
4. Click "Predict Match Outcome" (or follow CLI prompts)
5. View the predicted match outcome

## Future Enhancements

The web-based Streamlit version is designed to be extendable for future visualizations:

- Player performance statistics and visualization
- Team comparison charts
- Historical match result graphs
- Player contribution analysis

## Troubleshooting

### Streamlit Not Available

If Streamlit is not installed:
1. Install it with: `pip install streamlit`
2. Make sure you're in the virtual environment: `source afl_venv_3_9/bin/activate`

### Tkinter Not Available

If you see an error about Tkinter not being available, you can:
1. Use the web-based version instead: `streamlit run gui/predictor_streamlit.py`
2. Use the command-line version: `python gui/predictor_cli.py`
3. Install Tkinter for your Python version:
   - On macOS: `brew install python-tk@3.9`
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On Windows: Tkinter is usually included with Python installations

### TensorFlow Not Available

If TensorFlow is not available:
1. Install it with: `pip install tensorflow`
2. Make sure you're in the virtual environment with: `source afl_venv_3_9/bin/activate`

### Model File Not Found

If the model file is not found:
1. Make sure the model file exists at `model/output/model.h5`
2. If it doesn't exist, you may need to train the model first using the scripts in the `model` directory 