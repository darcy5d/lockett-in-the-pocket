# Lockett In The Pocket

This repository contains code for the "Lockett In The Pocket" project, which analyzes Australian Football League (AFL) data.

## Data Source

The AFL data is sourced from the [akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) repository, which provides comprehensive data for the Australian Football League, including:

- Match scores data from 1897 to 2025
- Player statistics for over 5,700 players
- Historical odds data from 2009 to 2024

## Setup

1. Create and activate the virtual environment:
```bash
python -m venv afl_venv
source afl_venv/bin/activate  # On Windows: afl_venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Fetching AFL Data

To download the latest AFL data from the source repository, run:
```bash
python fetch_afl_data.py
```

This will download the data into the `afl_data` directory, organized into subdirectories matching the source repository's structure.

## Using the AFL Data

After downloading the data, you can use the `load_afl_data.py` script to easily access and work with the data:

```bash
# To view available data files
python load_afl_data.py

# To use the data in your Python scripts
from load_afl_data import AFLDataLoader

# Initialize the loader
loader = AFLDataLoader()

# Load match data for a specific year
matches_2023 = loader.load_matches(2023)

# Load all match data
all_matches = loader.load_matches()

# Load data for a specific player
bontempelli_data = loader.load_player_data('Bontempelli')

# Load historical odds data
odds_data = loader.load_odds_data()
```

## Project Structure

- `fetch_afl_data.py`: Script to download the latest AFL data
- `load_afl_data.py`: Script to load and explore the downloaded AFL data
- `requirements.txt`: List of required Python packages
- `afl_data/`: Directory containing the downloaded AFL data
  - `data/`: Match and player statistics
  - `odds_data/`: Historical betting odds information 