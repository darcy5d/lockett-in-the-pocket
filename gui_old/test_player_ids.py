#!/usr/bin/env python3
import json
from pathlib import Path
import os
import pandas as pd
import glob

def load_player_names():
    # This will collect player names from data files
    player_names = {}
    player_ids = set()
    
    # Use glob to find all player files
    player_performance_files = glob.glob('../afl_data/data/players/*_performance_details.csv')
    personal_details_files = glob.glob('../afl_data/data/players/*_personal_details.csv')
    
    print(f"Found {len(player_performance_files)} player performance files")
    print(f"Found {len(personal_details_files)} personal details files")
    
    # Process performance files to get player IDs
    for file in player_performance_files:
        # Extract player ID from filename 
        player_id = os.path.basename(file).split('_performance')[0]
        player_ids.add(player_id)
    
    # Create player index lookup with IDs
    player_id_to_index = {player_id: i+1 for i, player_id in enumerate(sorted(player_ids))}
    player_id_to_index['unknown'] = 0
    
    # Process personal details files to get player names
    for file in personal_details_files:
        player_id = os.path.basename(file).split('_personal')[0]
        if player_id in player_id_to_index:
            try:
                df = pd.read_csv(file)
                if not df.empty and 'first_name' in df.columns and 'surname' in df.columns:
                    first_name = df.iloc[0]['first_name']
                    surname = df.iloc[0]['surname']
                    full_name = f"{first_name} {surname}"
                    player_names[full_name] = player_id
            except Exception as e:
                print(f"Error loading player details from {file}: {e}")
    
    # Show some stats
    print(f"Loaded {len(player_ids)} player IDs")
    print(f"Created {len(player_names)} player name mappings")
    
    return player_id_to_index, player_names

def load_player_index():
    # Load the player index from the model output
    player_index_path = Path('../model/output/player_index.json')
    
    if player_index_path.exists():
        with open(player_index_path, 'r') as f:
            player_index = json.load(f)
        print(f"Loaded {len(player_index)} player indices from {player_index_path}")
        return player_index
    else:
        print(f"Player index file not found at {player_index_path}")
        return {}

def test_player_lookup():
    # Load player data
    player_id_to_index, player_names = load_player_names()
    player_index = load_player_index()
    
    # Display the first 5 entries from player_names
    print("\nSample player names (name -> ID):")
    for name, player_id in list(player_names.items())[:5]:
        print(f"  {name} -> {player_id}")
    
    # Display the first 5 entries from player_index
    print("\nSample player indices (ID -> index):")
    for player_id, index in list(player_index.items())[:5]:
        print(f"  {player_id} -> {index}")
    
    # Test lookup for a few sample players
    test_names = list(player_names.keys())[:5]
    
    print("\nTesting player lookup for sample players:")
    for name in test_names:
        player_id = player_names.get(name, "unknown")
        player_idx = player_index.get(player_id, 0)
        print(f"  {name} -> ID: {player_id} -> Index: {player_idx}")
    
    # Interactive test
    print("\nInteractive player lookup test:")
    while True:
        name = input("Enter a player name (or 'q' to quit): ")
        if name.lower() == 'q':
            break
        
        player_id = player_names.get(name, "unknown")
        if player_id == "unknown":
            print(f"  Player '{name}' not found in player names dictionary")
            continue
            
        player_idx = player_index.get(player_id, 0)
        if player_idx == 0:
            print(f"  Player ID '{player_id}' not found in player index")
        else:
            print(f"  {name} -> ID: {player_id} -> Index: {player_idx}")

if __name__ == "__main__":
    test_player_lookup() 