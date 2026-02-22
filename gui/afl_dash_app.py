import sys
from pathlib import Path

# Add project root for core imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dash
from dash import dcc, html, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime

from core.mappings import PlayerMapper
from model.prediction_api import predict_match_outcome

# --- Utility functions for teams, grounds, and lineups ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LINEUP_DIR = os.path.join(PROJECT_ROOT, 'afl_data/data/lineups')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model/output')

# Get all teams from lineup files
def get_teams():
    lineup_files = glob.glob(os.path.join(LINEUP_DIR, 'team_lineups_*.csv'))
    teams = [os.path.basename(f).replace('team_lineups_', '').replace('.csv', '') for f in lineup_files]
    return sorted(teams)

# Get all grounds from match files
def get_grounds():
    match_files = glob.glob(os.path.join(PROJECT_ROOT, 'afl_data/data/matches/matches_*.csv'))
    grounds = set()
    for file in match_files:
        df = pd.read_csv(file)
        if 'venue' in df.columns:
            grounds.update(df['venue'].unique())
    return sorted(list(grounds))

# --- Helper functions for lineups and players ---
def get_last_lineup(team_name):
    file_path = os.path.join(LINEUP_DIR, f'team_lineups_{team_name}.csv')
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path)
    if len(df) == 0:
        return []
    latest_year = df['year'].max()
    df_year = df[df['year'] == latest_year]
    try:
        df_year.loc[:, 'round_num_numeric'] = pd.to_numeric(df_year['round_num'], errors='coerce')
        latest_round = df_year['round_num_numeric'].max()
        latest_lineup = df_year[df_year['round_num_numeric'] == latest_round].iloc[-1]
    except Exception:
        latest_lineup = df_year.iloc[-1]
    players_str = latest_lineup['players']
    if pd.isna(players_str):
        return []
    players = [p.strip() for p in players_str.split(';') if p.strip()]
    return players

def get_team_players_this_season(team_name):
    year = datetime.now().year
    file_path = os.path.join(LINEUP_DIR, f'team_lineups_{team_name}.csv')
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path)
    df = df[df['year'] == year]
    players = set()
    if 'players' in df.columns:
        for player_str in df['players']:
            if pd.notna(player_str):
                players.update([p.strip() for p in player_str.split(';') if p.strip()])
    return sorted(list(players), key=lambda x: format_player_name(x).lower())

def format_player_name(player_id):
    # Convert LastName_FirstName_12345678 to FirstName LastName
    parts = player_id.split('_')
    if len(parts) >= 2:
        return f"{parts[1]} {parts[0]}"
    return player_id

def get_all_players():
    with open(os.path.join(MODEL_DIR, 'player_index.json'), 'r') as f:
        player_index = json.load(f)
    # Create a dictionary of formatted names to original IDs
    formatted_to_original = {format_player_name(p): p for p in player_index.keys()}
    # Sort the formatted names alphabetically
    sorted_names = sorted(formatted_to_original.keys(), key=str.lower)
    # Return a dictionary with sorted names mapping to original IDs
    return {name: formatted_to_original[name] for name in sorted_names}

# --- Lineup UI components ---
def lineup_chips(players, col_key):
    # Sort players by their formatted names
    sorted_players = sorted(players, key=lambda x: format_player_name(x).lower())
    return html.Div([
        dbc.Badge(
            format_player_name(player),
            color='secondary',
            className='mx-1',
            pill=True,
            style={'fontSize': '1em', 'margin': '2px', 'cursor': 'pointer'},
            id={'type': f'remove-{col_key}-player', 'player_id': player}
        ) for player in sorted_players
    ], style={'minHeight': '50px'})

def addable_players_list(players, col_key):
    # Players are already sorted in get_all_players and get_team_players_this_season
    return html.Div([
        dbc.Button(
            player,
            color='success',
            size='sm',
            className='my-1',
            id={'type': f'add-{col_key}-player', 'player_id': player}
        ) for player in players
    ], style={'maxHeight': '200px', 'overflowY': 'auto'})

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'AFL Match Predictor'

# --- Initial Data ---
TEAMS = get_teams()
GROUNDS = get_grounds()
PLAYER_MAPPER = PlayerMapper()


def _to_player_ids(store_values):
    """Convert store values (display names or player_ids) to player_ids for prediction."""
    if not store_values:
        return []
    ids = []
    for v in store_values:
        if isinstance(v, str) and "_" in v and len(v.split("_")[-1]) == 8 and v.split("_")[-1].isdigit():
            ids.append(v)
        else:
            pid = PLAYER_MAPPER.to_player_id(v)
            if pid and pid != "unknown":
                ids.append(pid)
    return ids

# --- Layout ---
app.layout = dbc.Container([
    # Store components for persisting lineups and season players
    dcc.Store(id='home-lineup-store', data=[]),
    dcc.Store(id='away-lineup-store', data=[]),
    dcc.Store(id='home-season-players-store', data=[]),
    dcc.Store(id='away-season-players-store', data=[]),
    
    html.H1('AFL MATCH PREDICTOR', className='my-4'),
    dbc.Row([
        dbc.Col([
            html.Label('Home team'),
            dcc.Dropdown(id='home-team-dropdown', options=[{'label': t.title(), 'value': t} for t in TEAMS], value=TEAMS[0] if TEAMS else None),
        ], width=4),
        dbc.Col([
            html.Label('Away team'),
            dcc.Dropdown(id='away-team-dropdown'),
        ], width=4),
        dbc.Col([
            html.Label('Ground'),
            dcc.Dropdown(id='ground-dropdown', options=[{'label': g, 'value': g} for g in GROUNDS], value=GROUNDS[0] if GROUNDS else None),
        ], width=4),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H5('GAME LINEUP', className='mb-2'),
            dcc.Checklist(id='home-lineup-checklist', inputStyle={"margin-right": "8px"}, style={"maxHeight": "150px", "overflowY": "auto"}),
            dbc.Button('Remove from game lineup', id='remove-from-game-lineup-btn', color='danger', className='my-2'),
            html.H6('CURRENT SEASON PLAYERS', className='mt-3'),
            dcc.Checklist(id='home-season-players-checklist', inputStyle={"margin-right": "8px"}, style={"maxHeight": "150px", "overflowY": "auto"}),
            dbc.Button('Promote to game day', id='promote-to-game-day-btn', color='success', className='my-2'),
            html.H6('Search Players', className='mt-3'),
            dbc.InputGroup([
                dcc.Input(id='home-player-search', type='text', placeholder='Search players...', className='form-control'),
                dbc.Button('Search', id='home-player-search-btn', color='secondary', n_clicks=0)
            ], className='mb-2'),
            dcc.Dropdown(id='home-search-results-dropdown', multi=True, placeholder='Select players to draft...'),
            dbc.Button('Draft to season list', id='draft-to-season-list-btn', color='primary', className='my-2'),
        ], width=6),
        dbc.Col([
            html.H5('GAME LINEUP', className='mb-2'),
            dcc.Checklist(id='away-lineup-checklist', inputStyle={"margin-right": "8px"}, style={"maxHeight": "150px", "overflowY": "auto"}),
            dbc.Button('Remove from game lineup', id='remove-from-away-game-lineup-btn', color='danger', className='my-2'),
            html.H6('CURRENT SEASON PLAYERS', className='mt-3'),
            dcc.Checklist(id='away-season-players-checklist', inputStyle={"margin-right": "8px"}, style={"maxHeight": "150px", "overflowY": "auto"}),
            dbc.Button('Promote to game day', id='promote-to-away-game-day-btn', color='success', className='my-2'),
            html.H6('Search Players', className='mt-3'),
            dbc.InputGroup([
                dcc.Input(id='away-player-search', type='text', placeholder='Search players...', className='form-control'),
                dbc.Button('Search', id='away-player-search-btn', color='secondary', n_clicks=0)
            ], className='mb-2'),
            dcc.Dropdown(id='away-search-results-dropdown', multi=True, placeholder='Select players to draft...'),
            dbc.Button('Draft to season list', id='draft-to-away-season-list-btn', color='primary', className='my-2'),
        ], width=6),
    ]),
    html.Hr(),
    dbc.Button('Predict Match Outcome', id='predict-btn', color='primary', className='my-3'),
    html.Div(id='prediction-results'),
], fluid=True)


# --- Predict callback ---
@app.callback(
    Output('prediction-results', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('home-lineup-store', 'data'),
    State('away-lineup-store', 'data'),
    State('home-team-dropdown', 'value'),
    State('away-team-dropdown', 'value'),
    prevent_initial_call=True
)
def run_prediction(n_clicks, home_lineup, away_lineup, home_team, away_team):
    if not n_clicks:
        return ""
    home_ids = _to_player_ids(home_lineup or [])
    away_ids = _to_player_ids(away_lineup or [])
    if len(home_ids) < 18 or len(away_ids) < 18:
        return dbc.Alert(
            "Both teams need at least 18 players selected.",
            color="warning",
            className="my-3"
        )
    result = predict_match_outcome(home_ids, away_ids)
    if "error" in result:
        return dbc.Alert(f"Prediction error: {result['error']}", color="danger", className="my-3")
    home_name = home_team or "Home"
    away_name = away_team or "Away"
    home_score = result["team1"]["goals"] * 6 + result["team1"]["behinds"]
    away_score = result["team2"]["goals"] * 6 + result["team2"]["behinds"]
    probs = result["winner_probabilities"]
    return dbc.Card([
        dbc.CardHeader("Prediction Results"),
        dbc.CardBody([
            html.Hr(),
            html.H6("Win probabilities"),
            dbc.Row([
                dbc.Col(html.Div([html.Strong(home_name), html.Br(), f"{probs['team1_win']*100:.1f}%"])),
                dbc.Col(html.Div(["Draw", html.Br(), f"{probs['draw']*100:.1f}%"])),
                dbc.Col(html.Div([html.Strong(away_name), html.Br(), f"{probs['team2_win']*100:.1f}%"])),
            ]),
            html.Hr(),
            html.H6("Predicted scores"),
            dbc.Row([
                dbc.Col(html.Div([html.Strong(home_name), html.Br(),
                    f"{result['team1']['goals']:.1f}.{result['team1']['behinds']:.1f} ({home_score:.0f} pts)"])),
                dbc.Col(html.Div([html.Strong(away_name), html.Br(),
                    f"{result['team2']['goals']:.1f}.{result['team2']['behinds']:.1f} ({away_score:.0f} pts)"])),
            ]),
            html.Hr(),
            html.P(f"Predicted margin: {abs(result['margin']):.1f} points ({home_name if result['margin'] > 0 else away_name} by)"),
        ]),
    ], className="my-3")

# --- Callbacks for lineup management ---
@app.callback(
    Output('home-lineup-store', 'data'),
    Output('away-lineup-store', 'data'),
    Output('home-season-players-store', 'data'),
    Output('away-season-players-store', 'data'),
    Input('home-team-dropdown', 'value'),
    Input('away-team-dropdown', 'value'),
    prevent_initial_call=True
)
def initialize_lineups_and_season_players(home_team, away_team):
    home_players = get_last_lineup(home_team) if home_team else []
    away_players = get_last_lineup(away_team) if away_team else []
    home_season_players = get_team_players_this_season(home_team) if home_team else []
    away_season_players = get_team_players_this_season(away_team) if away_team else []
    # Remove lineup players from season players (so they only appear in one section)
    home_season_players = [p for p in home_season_players if p not in home_players]
    away_season_players = [p for p in away_season_players if p not in away_players]
    return home_players, away_players, home_season_players, away_season_players

@app.callback(
    Output('home-lineup-store', 'data', allow_duplicate=True),
    Output('home-season-players-store', 'data', allow_duplicate=True),
    Output('home-lineup-checklist', 'options'),
    Output('home-lineup-checklist', 'value'),
    Output('home-season-players-checklist', 'options'),
    Output('home-season-players-checklist', 'value'),
    Input('remove-from-game-lineup-btn', 'n_clicks'),
    State('home-lineup-checklist', 'value'),
    State('home-lineup-store', 'data'),
    State('home-season-players-store', 'data'),
    prevent_initial_call=True
)
def remove_from_game_lineup(n_clicks, selected_players, lineup, season_players):
    if not n_clicks or not selected_players:
        options_lineup = [{'label': format_player_name(p), 'value': p} for p in sorted(lineup, key=lambda x: format_player_name(x).lower())]
        options_season = [{'label': format_player_name(p), 'value': p} for p in sorted(season_players, key=lambda x: format_player_name(x).lower())]
        return lineup, season_players, options_lineup, [], options_season, []
    new_lineup = [p for p in lineup if p not in selected_players]
    new_season_players = season_players + [p for p in selected_players if p not in season_players]
    new_season_players = [p for p in new_season_players if p not in new_lineup]
    options_lineup = [{'label': format_player_name(p), 'value': p} for p in sorted(new_lineup, key=lambda x: format_player_name(x).lower())]
    options_season = [{'label': format_player_name(p), 'value': p} for p in sorted(new_season_players, key=lambda x: format_player_name(x).lower())]
    return new_lineup, new_season_players, options_lineup, [], options_season, []

@app.callback(
    Output('home-lineup-store', 'data', allow_duplicate=True),
    Output('home-season-players-store', 'data', allow_duplicate=True),
    Output('home-lineup-checklist', 'options'),
    Output('home-lineup-checklist', 'value'),
    Output('home-season-players-checklist', 'options'),
    Output('home-season-players-checklist', 'value'),
    Input('promote-to-game-day-btn', 'n_clicks'),
    State('home-season-players-checklist', 'value'),
    State('home-lineup-store', 'data'),
    State('home-season-players-store', 'data'),
    prevent_initial_call=True
)
def promote_to_game_day(n_clicks, selected_players, lineup, season_players):
    if not n_clicks or not selected_players:
        options_lineup = [{'label': format_player_name(p), 'value': p} for p in sorted(lineup, key=lambda x: format_player_name(x).lower())]
        options_season = [{'label': format_player_name(p), 'value': p} for p in sorted(season_players, key=lambda x: format_player_name(x).lower())]
        return lineup, season_players, options_lineup, [], options_season, []
    new_season_players = [p for p in season_players if p not in selected_players]
    new_lineup = lineup + [p for p in selected_players if p not in lineup]
    new_lineup = [p for p in new_lineup if p not in new_season_players]
    options_lineup = [{'label': format_player_name(p), 'value': p} for p in sorted(new_lineup, key=lambda x: format_player_name(x).lower())]
    options_season = [{'label': format_player_name(p), 'value': p} for p in sorted(new_season_players, key=lambda x: format_player_name(x).lower())]
    return new_lineup, new_season_players, options_lineup, [], options_season, []

@app.callback(
    Output('home-season-players-store', 'data', allow_duplicate=True),
    Output('home-season-players-checklist', 'options'),
    Output('home-season-players-checklist', 'value'),
    Output('home-search-results-dropdown', 'options'),
    Output('home-search-results-dropdown', 'value'),
    Input('draft-to-season-list-btn', 'n_clicks'),
    State('home-search-results-dropdown', 'value'),
    State('home-season-players-store', 'data'),
    State('home-lineup-store', 'data'),
    State('home-player-search', 'value'),
    prevent_initial_call=True
)
def draft_to_season_list(n_clicks, selected_players, season_players, lineup, search_value):
    if not n_clicks or not selected_players:
        options_season = [{'label': format_player_name(p), 'value': p} for p in sorted(season_players, key=lambda x: format_player_name(x).lower())]
        all_players = get_all_players()
        available_players = [p for p in all_players.keys() if all_players[p] not in season_players and all_players[p] not in lineup]
        filtered_players = [p for p in available_players if search_value and search_value.lower() in p.lower()]
        options_search = [{'label': p, 'value': all_players[p]} for p in sorted(filtered_players, key=str.lower)]
        return season_players, options_season, [], options_search, []
    new_season_players = season_players + [p for p in selected_players if p not in season_players and p not in lineup]
    new_season_players = [p for p in new_season_players if p not in lineup]
    options_season = [{'label': format_player_name(p), 'value': p} for p in sorted(new_season_players, key=lambda x: format_player_name(x).lower())]
    all_players = get_all_players()
    available_players = [p for p in all_players.keys() if all_players[p] not in new_season_players and all_players[p] not in lineup]
    filtered_players = [p for p in available_players if search_value and search_value.lower() in p.lower()]
    options_search = [{'label': p, 'value': all_players[p]} for p in sorted(filtered_players, key=str.lower)]
    return new_season_players, options_season, [], options_search, []

@app.callback(
    Output('away-lineup-store', 'data', allow_duplicate=True),
    Output('away-season-players-store', 'data', allow_duplicate=True),
    Input({'type': 'remove-away-player', 'player_id': ALL}, 'n_clicks'),
    Input({'type': 'add-away-player', 'player_id': ALL}, 'n_clicks'),
    State('away-lineup-store', 'data'),
    State('away-season-players-store', 'data'),
    prevent_initial_call=True
)
def update_away_lineup_and_season_players(remove_clicks, add_clicks, lineup, season_players):
    if not ctx.triggered:
        return lineup, season_players
    triggered = ctx.triggered[0]
    triggered_id = json.loads(triggered['prop_id'].split('.')[0])
    player_id = triggered_id.get('player_id')
    lineup = [p for p in lineup if p != player_id]
    season_players = [p for p in season_players if p != player_id]
    if 'remove' in triggered_id['type']:
        season_players.append(player_id)
    elif 'add' in triggered_id['type']:
        lineup.append(player_id)
    return lineup, season_players

@app.callback(
    Output('home-season-players-store', 'data', allow_duplicate=True),
    Input({'type': 'add-home-player', 'player_id': ALL}, 'n_clicks'),
    State('home-season-players-store', 'data'),
    State('home-lineup-store', 'data'),
    prevent_initial_call=True
)
def add_home_search_player_to_season(add_clicks, season_players, lineup):
    if not ctx.triggered:
        return season_players
    triggered = ctx.triggered[0]
    triggered_id = json.loads(triggered['prop_id'].split('.')[0])
    player_id = triggered_id.get('player_id')
    # Only add if not already in either list
    if player_id and player_id not in season_players and player_id not in lineup:
        # Remove from lineup if present (shouldn't be, but for safety)
        if player_id in lineup:
            lineup.remove(player_id)
        season_players.append(player_id)
    # Remove from season_players if in lineup (shouldn't be, but for safety)
    season_players = [p for p in season_players if p not in lineup]
    return season_players

@app.callback(
    Output('away-season-players-store', 'data', allow_duplicate=True),
    Input({'type': 'add-away-player', 'player_id': ALL}, 'n_clicks'),
    State('away-season-players-store', 'data'),
    State('away-lineup-store', 'data'),
    prevent_initial_call=True
)
def add_away_search_player_to_season(add_clicks, season_players, lineup):
    if not ctx.triggered:
        return season_players
    triggered = ctx.triggered[0]
    triggered_id = json.loads(triggered['prop_id'].split('.')[0])
    player_id = triggered_id.get('player_id')
    if player_id and player_id not in season_players and player_id not in lineup:
        if player_id in lineup:
            lineup.remove(player_id)
        season_players.append(player_id)
    season_players = [p for p in season_players if p not in lineup]
    return season_players

# --- Callbacks to update away team and ground dropdowns ---
@app.callback(
    Output('away-team-dropdown', 'options'),
    Output('away-team-dropdown', 'value'),
    Input('home-team-dropdown', 'value')
)
def update_away_team_options(home_team):
    options = [{'label': t.title(), 'value': t} for t in TEAMS if t != home_team]
    value = options[0]['value'] if options else None
    return options, value

@app.callback(
    Output('home-lineup-checklist', 'options'),
    Output('home-lineup-checklist', 'value'),
    Input('home-lineup-store', 'data'),
    prevent_initial_call=True
)
def populate_home_lineup_checklist(lineup):
    options = [{'label': format_player_name(p), 'value': p} for p in sorted(lineup, key=lambda x: format_player_name(x).lower())]
    return options, lineup

@app.callback(
    Output('home-season-players-checklist', 'options'),
    Output('home-season-players-checklist', 'value'),
    Input('home-season-players-store', 'data'),
    prevent_initial_call=True
)
def populate_home_season_players_checklist(season_players):
    options = [{'label': format_player_name(p), 'value': p} for p in sorted(season_players, key=lambda x: format_player_name(x).lower())]
    return options, []

@app.callback(
    Output('home-search-results-dropdown', 'options'),
    Output('home-search-results-dropdown', 'value'),
    Input('home-player-search-btn', 'n_clicks'),
    State('home-player-search', 'value'),
    State('home-season-players-store', 'data'),
    State('home-lineup-store', 'data'),
    prevent_initial_call=True
)
def populate_home_search_results_dropdown(n_clicks, search_value, season_players, lineup):
    if not search_value:
        return [], []
    all_players = get_all_players()
    available_players = [p for p in all_players.keys() if all_players[p] not in season_players and all_players[p] not in lineup]
    filtered_players = [p for p in available_players if search_value.lower() in p.lower()]
    options = [{'label': p, 'value': all_players[p]} for p in sorted(filtered_players, key=str.lower)]
    return options, []

if __name__ == '__main__':
    app.run(debug=True, port=8051) 