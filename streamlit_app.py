import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import os
import glob
from tensorflow.keras.models import load_model
import json
import joblib
from datetime import datetime

# Custom loss function
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Custom bounded output layer
@register_keras_serializable()
class BoundedOutputLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that bounds the output between a minimum and maximum value.
    Uses sigmoid activation and scales the output to the desired range.
    """
    def __init__(self, min_value, max_value, **kwargs):
        super(BoundedOutputLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        
    def build(self, input_shape):
        super(BoundedOutputLayer, self).build(input_shape)
        
    def call(self, inputs):
        scaled_sigmoid = tf.sigmoid(inputs)
        return self.min_value + (self.max_value - self.min_value) * scaled_sigmoid
        
    def get_config(self):
        config = super(BoundedOutputLayer, self).get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        return config
        
    def compute_output_shape(self, input_shape):
        return input_shape

# Custom pooling layer with explicit output shape
@register_keras_serializable()
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
        config.update({'output_dim': self.output_dim})
        return config

# Register the custom objects
tf.keras.utils.get_custom_objects()['mse'] = mse
tf.keras.utils.get_custom_objects()['BoundedOutputLayer'] = BoundedOutputLayer
tf.keras.utils.get_custom_objects()['MeanPoolingLayer'] = MeanPoolingLayer

# Set page config
st.set_page_config(page_title="AFL Match Predictor", layout="wide")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
LINEUP_DIR = os.path.join(PROJECT_ROOT, 'afl_data/data/lineups')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model/output')
MAX_PLAYERS_PER_TEAM = 23  # 18 + 5 subs for 2025+

@st.cache_resource
def load_resources():
    model = load_model(os.path.join(MODEL_DIR, 'model.h5'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    with open(os.path.join(MODEL_DIR, 'player_index.json'), 'r') as f:
        player_index = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'r') as f:
        feature_cols = json.load(f)
    return model, scaler, player_index, feature_cols

def get_teams():
    lineup_files = glob.glob(os.path.join(LINEUP_DIR, 'team_lineups_*.csv'))
    teams = [os.path.basename(f).replace('team_lineups_', '').replace('.csv', '') for f in lineup_files]
    return sorted(teams)

@st.cache_data
def get_grounds():
    match_files = glob.glob(os.path.join(PROJECT_ROOT, 'afl_data/data/matches/matches_*.csv'))
    grounds = set()
    for file in match_files:
        df = pd.read_csv(file)
        if 'ground' in df.columns:
            grounds.update(df['ground'].unique())
    return sorted(list(grounds))

@st.cache_data
def get_last_lineup(team_name):
    file_path = os.path.join(LINEUP_DIR, f'team_lineups_{team_name}.csv')
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path)
    if len(df) == 0:
        return []
    latest_year = df['year'].max()
    # Ensure round_num is treated as numeric (in case of finals, etc.)
    df_year = df[df['year'] == latest_year]
    # Try to convert round_num to numeric, fallback to string if needed
    try:
        df_year['round_num_numeric'] = pd.to_numeric(df_year['round_num'], errors='coerce')
        latest_round = df_year['round_num_numeric'].max()
        latest_lineup = df_year[df_year['round_num_numeric'] == latest_round].iloc[-1]
    except Exception:
        latest_lineup = df_year.iloc[-1]
    players_str = latest_lineup['players']
    if pd.isna(players_str):
        return []
    players = [p.strip() for p in players_str.split(';') if p.strip()]
    return players

@st.cache_data
def get_team_players_this_season(team_name):
    year = datetime.now().year
    file_path = os.path.join(LINEUP_DIR, f'team_lineups_{team_name}.csv')
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path)
    df = df[df['year'] == year]
    players = set()
    for col in df.columns:
        if col.startswith('player') and col != 'player_count':
            players.update(df[col].dropna().unique())
    return sorted(list(players))

@st.cache_data
def get_all_players():
    with open(os.path.join(MODEL_DIR, 'player_index.json'), 'r') as f:
        player_index = json.load(f)
    return sorted(list(player_index.keys()))

def prepare_match_features(home_team, away_team, ground, feature_cols):
    # Create a DataFrame with the match features
    match_features = pd.DataFrame(columns=feature_cols)
    
    # Set match type to regular season
    match_features['match_type_Regular'] = 1
    for col in feature_cols:
        if col.startswith('match_type_') and col != 'match_type_Regular':
            match_features[col] = 0
    
    # Add ground information
    match_features[f'ground_{ground}'] = 1
    for col in feature_cols:
        if col.startswith('ground_') and col != f'ground_{ground}':
            match_features[col] = 0
    
    # Add team information
    match_features[f'team_1_team_name_{home_team}'] = 1
    match_features[f'team_2_team_name_{away_team}'] = 1
    for col in feature_cols:
        if col.startswith('team_1_team_name_') and col != f'team_1_team_name_{home_team}':
            match_features[col] = 0
        if col.startswith('team_2_team_name_') and col != f'team_2_team_name_{away_team}':
            match_features[col] = 0
    
    return match_features

def prepare_player_indices(players, player_index):
    # Convert player names to indices
    indices = [player_index.get(player, 0) for player in players]  # 0 for unknown players
    
    # Pad or truncate to MAX_PLAYERS_PER_TEAM
    if len(indices) < MAX_PLAYERS_PER_TEAM:
        indices.extend([0] * (MAX_PLAYERS_PER_TEAM - len(indices)))
    elif len(indices) > MAX_PLAYERS_PER_TEAM:
        indices = indices[:MAX_PLAYERS_PER_TEAM]
    
    return np.array(indices)

def make_prediction(model, scaler, match_features, home_player_indices, away_player_indices):
    # Scale match features
    match_features_scaled = scaler.transform(match_features)
    
    # Prepare input data
    inputs = [
        match_features_scaled,
        home_player_indices.reshape(1, -1),
        away_player_indices.reshape(1, -1),
        np.zeros((1, 21)),  # Placeholder for enhanced features
        np.zeros((1, 21))   # Placeholder for enhanced features
    ]
    
    # Make prediction
    predictions = model.predict(inputs)
    
    # Process predictions
    winner_probs = predictions[0][0]  # [team1_win, draw, team2_win]
    team1_goals = predictions[1][0][0]
    team1_behinds = predictions[2][0][0]
    team2_goals = predictions[3][0][0]
    team2_behinds = predictions[4][0][0]
    margin = predictions[5][0][0]
    
    return {
        'winner_probs': winner_probs,
        'team1_goals': team1_goals,
        'team1_behinds': team1_behinds,
        'team2_goals': team2_goals,
        'team2_behinds': team2_behinds,
        'margin': margin
    }

def main():
    st.title("AFL MATCH PREDICTOR")
    model, scaler, player_index, feature_cols = load_resources()
    teams = get_teams()
    grounds = get_grounds()

    # Top: dropdowns for home, away, ground
    col_top1, col_top2, col_top3 = st.columns([2,2,2])
    with col_top1:
        home_team = st.selectbox("Home team", teams, key="home_team")
    with col_top2:
        away_team = st.selectbox("Away team", [t for t in teams if t != home_team], key="away_team")
    with col_top3:
        ground = st.selectbox("Ground", grounds, key="ground")

    # Two columns for lineups
    col1, col2 = st.columns(2)
    for col, team, col_key in zip([col1, col2], [home_team, away_team], ["home", "away"]):
        with col:
            st.markdown(f"**{team} lineup**")
            last_lineup = get_last_lineup(team)
            # Use session state to persist changes
            if f"selected_{col_key}_players" not in st.session_state:
                st.session_state[f"selected_{col_key}_players"] = last_lineup.copy()
            selected_players = st.session_state[f"selected_{col_key}_players"]
            # Show current lineup with remove buttons
            for i, player in enumerate(selected_players):
                cols = st.columns([8,1])
                cols[0].write(player)
                if cols[1].button("❌", key=f"remove_{col_key}_{i}"):
                    selected_players.pop(i)
                    st.experimental_rerun()
            # Add section: other players this season
            st.markdown("other player available who played in this season")
            season_players = get_team_players_this_season(team)
            available_players = [p for p in season_players if p not in selected_players]
            for p in available_players:
                cols = st.columns([8,1])
                cols[0].write(p)
                if cols[1].button("➕", key=f"add_season_{col_key}_{p}"):
                    selected_players.append(p)
                    st.experimental_rerun()
            # Add section: player database search
            st.markdown("Player database search")
            search = st.text_input(f"type name here, e.g. Todd", key=f"search_{col_key}")
            if search:
                all_players = get_all_players()
                matches = [p for p in all_players if search.lower() in p.lower() and p not in selected_players]
                for p in matches:
                    cols = st.columns([8,1])
                    cols[0].write(p)
                    if cols[1].button("➕", key=f"add_db_{col_key}_{p}"):
                        selected_players.append(p)
                        st.experimental_rerun()

    # Prediction button and results (can be below both columns)
    st.markdown("---")
    if st.button("Predict Match Outcome"):
        home_players = st.session_state["selected_home_players"]
        away_players = st.session_state["selected_away_players"]
        if len(home_players) < 18 or len(away_players) < 18:
            st.error("Both teams must have at least 18 players selected")
        else:
            match_features = prepare_match_features(home_team, away_team, ground, feature_cols)
            home_player_indices = prepare_player_indices(home_players, player_index)
            away_player_indices = prepare_player_indices(away_players, player_index)
            prediction = make_prediction(model, scaler, match_features, home_player_indices, away_player_indices)
            st.header("Prediction Results")
            winner_probs = prediction['winner_probs']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{home_team} Win", f"{winner_probs[0]*100:.1f}%")
            with col2:
                st.metric("Draw", f"{winner_probs[1]*100:.1f}%")
            with col3:
                st.metric(f"{away_team} Win", f"{winner_probs[2]*100:.1f}%")
            st.subheader("Predicted Scores")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    home_team,
                    f"{prediction['team1_goals']:.1f}.{prediction['team1_behinds']:.1f} ({prediction['team1_goals']*6 + prediction['team1_behinds']:.1f})"
                )
            with col2:
                st.metric(
                    away_team,
                    f"{prediction['team2_goals']:.1f}.{prediction['team2_behinds']:.1f} ({prediction['team2_goals']*6 + prediction['team2_behinds']:.1f})"
                )
            st.subheader("Predicted Margin")
            st.metric(
                "Margin",
                f"{abs(prediction['margin']):.1f} points",
                f"{home_team if prediction['margin'] > 0 else away_team} by"
            )

if __name__ == "__main__":
    main() 