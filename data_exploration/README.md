# AFL Data Exploration

This directory contains notebooks and scripts for exploratory data analysis of AFL (Australian Football League) data.

## Purpose

The purpose of this directory is to:

1. Explore and visualize AFL match data
2. Analyze player statistics and performance trends
3. Investigate betting odds data
4. Generate insights into the AFL data

## Contents

- `basic_stats.py`: Simple script for generating basic statistics about matches and players
- `match_analysis.ipynb`: Jupyter notebook for analyzing match data
- `player_analysis.ipynb`: Jupyter notebook for analyzing player statistics

## Usage

To use these notebooks and scripts, ensure you have first downloaded the AFL data using the scripts in the `datafetch` directory:

```bash
cd datafetch
python fetch_afl_data.py
```

Then you can either:

1. Run the basic stats script:
```bash
cd data_exploration
python basic_stats.py
```

2. Launch Jupyter Notebook to open and run the interactive notebooks:
```bash
cd data_exploration
jupyter notebook
```

## Required Packages

The data exploration scripts and notebooks require the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- jupyter

## Example Notebooks

- (Future) `match_analysis.ipynb`: Analysis of match results, scores, and trends
- (Future) `player_performance.ipynb`: Player performance analysis
- (Future) `betting_analysis.ipynb`: Analysis of historical betting odds 