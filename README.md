# Lockett In The Pocket

This repository contains code for the "Lockett In The Pocket" project, which analyzes Australian Football League (AFL) data.

## Project Structure

The project is organized into two main directories:

1. **datafetch**: Scripts for fetching and loading AFL data
2. **data_exploration**: Notebooks and scripts for exploring and analyzing the data

## Data Source

The AFL data is sourced from the [akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) repository, which provides comprehensive data for the Australian Football League, including:

- Match scores data from 1897 to 2025
- Player statistics for over 13,000 players
- Historical odds data from 2009 to 2024

## Setup

1. Clone the repository:
```bash
git clone https://github.com/darcy5d/lockett-in-the-pocket.git
cd lockett-in-the-pocket
```

2. Create and activate the virtual environment:
```bash
python -m venv afl_venv
source afl_venv/bin/activate  # On Windows: afl_venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Fetching AFL Data

To download the latest AFL data from the source repository, run:
```bash
cd datafetch
python fetch_afl_data.py
```

This will download the data into the `afl_data` directory in the project root, organized into subdirectories matching the source repository's structure.

## Using the AFL Data

After downloading the data, you can use the `load_afl_data.py` script to easily access and work with the data:

```python
from datafetch.load_afl_data import AFLDataLoader

# Initialize the loader
loader = AFLDataLoader()

# Load match data for a specific year
matches_2023 = loader.load_matches(2023)

# Load all match data
all_matches = loader.load_matches()

# Load data for a specific player
bontempelli_performance = loader.load_player_data('bontempelli', data_type='performance')
bontempelli_personal = loader.load_player_data('bontempelli', data_type='personal')

# Load historical odds data
odds_data = loader.load_odds_data()
```

## Data Exploration

The `data_exploration` directory contains scripts and notebooks for analyzing AFL data. To run basic statistics on the data:

```bash
cd data_exploration
python basic_stats.py
```

This will generate basic statistics for match data and player data, providing insights into the AFL dataset.

## Changes Made

- Moved `neural_network.py` to the `model` directory.
- Deleted `match_analysis.ipynb` and `player_analysis.ipynb` from the `data_exploration` directory.
- Excluded the `afl_data` folder from the repository as it is sourced externally. 