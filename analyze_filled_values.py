import pandas as pd
import numpy as np
import os

# Define file paths for data
match_files = [
    'afl_data/data/matches/matches_2017.csv',
    'afl_data/data/matches/matches_2018.csv',
    'afl_data/data/matches/matches_2019.csv',
    'afl_data/data/matches/matches_2020.csv',
    'afl_data/data/matches/matches_2021.csv',
    'afl_data/data/matches/matches_2022.csv',
    'afl_data/data/matches/matches_2023.csv'
]

def analyze_nan_filling():
    print("===== ANALYZING COLUMNS FILLED WITH MEAN VALUES =====")
    
    # Load match data from all existing files
    print("Loading match data...")
    match_dfs = []
    for f in match_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                match_dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    match_data = pd.concat(match_dfs)
    print(f"Combined match data shape: {match_data.shape}")
    
    # Convert goal and behind columns to numeric, properly handling NaN values
    print("\nConverting goals and behinds to numeric values...")
    for col in match_data.columns:
        if '_goals' in col or '_behinds' in col:
            match_data[col] = pd.to_numeric(match_data[col], errors='coerce')
    
    # Analyze missing values
    print("\n===== MISSING VALUES ANALYSIS =====")
    
    # Get numeric columns
    numeric_cols = match_data.select_dtypes(include=['number']).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric columns")
    
    # Calculate nan counts and percentages before filling
    total_rows = len(match_data)
    nan_counts = match_data[numeric_cols].isna().sum()
    nan_percent = (nan_counts / total_rows * 100).round(2)
    
    # Create DataFrame to display results, sorted by percentage
    missing_data = pd.DataFrame({
        'Column': nan_counts.index,
        'Missing Count': nan_counts.values,
        'Missing Percentage': nan_percent.values
    })
    missing_data = missing_data.sort_values('Missing Percentage', ascending=False)
    
    # Only show columns with NaN values
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    print("\nColumns with missing values (would be filled with mean):")
    if len(missing_data) > 0:
        print(missing_data.to_string(index=False))
        
        # Count rows that would be affected by any mean filling
        rows_with_nans = match_data[numeric_cols].isna().any(axis=1).sum()
        percent_rows_affected = (rows_with_nans / total_rows * 100).round(2)
        
        print(f"\nOut of {total_rows} total rows:")
        print(f"- {rows_with_nans} rows ({percent_rows_affected}%) have at least one numeric value that would be filled with mean")
        
        # Print mean values that would be used
        print("\nMean values that would be used for filling:")
        means = match_data[numeric_cols].mean()
        
        # Only show means for columns with NaN values
        cols_with_nans = missing_data['Column'].tolist()
        print(means[cols_with_nans].to_string())
    else:
        print("No missing values found in numeric columns!")
    
    print("\n===== ANALYSIS COMPLETE =====")

if __name__ == "__main__":
    analyze_nan_filling() 