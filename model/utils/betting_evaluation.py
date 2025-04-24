#!/usr/bin/env python3
"""
Betting Evaluation Utilities

This module provides functions for evaluating the performance of betting strategies
based on model predictions and historical odds data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('betting_evaluation')

def calculate_kelly_stake(probability: float, odds: float, fraction: float = 1.0) -> float:
    """
    Calculate the optimal stake using the Kelly Criterion.
    
    Args:
        probability: Estimated probability of winning
        odds: Decimal odds offered by the bookmaker
        fraction: Fraction of the full Kelly to use (for risk management)
        
    Returns:
        Optimal stake as a proportion of bankroll
    """
    # Kelly formula: f* = (bp - q) / b
    # where:
    # f* = fraction of bankroll to wager
    # b = decimal odds - 1 (i.e., the profit if you win)
    # p = probability of winning
    # q = probability of losing (1 - p)
    
    b = odds - 1  # Potential profit
    
    if b <= 0 or probability <= 0:
        return 0.0
    
    # Calculate full Kelly stake
    q = 1 - probability
    kelly = (b * probability - q) / b
    
    # Apply Kelly fraction and ensure non-negative
    return max(0, kelly * fraction)

def evaluate_betting_strategy(
    predictions: pd.DataFrame,
    odds_data: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    stake_method: str = 'flat',
    flat_stake: float = 10.0,
    kelly_fraction: float = 0.5,
    threshold: float = 0.0
) -> Dict:
    """
    Evaluate a betting strategy using model predictions and odds data.
    
    Args:
        predictions: DataFrame with match predictions (must have columns for match_id, pred_probability)
        odds_data: DataFrame with odds data (must have columns for match_id, odds, actual_result)
        initial_bankroll: Starting bankroll
        stake_method: Method for determining stake ('flat' or 'kelly')
        flat_stake: Amount to bet for flat staking
        kelly_fraction: Fraction of Kelly criterion to use
        threshold: Minimum edge required to place a bet
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating betting strategy...")
    
    # Merge predictions with odds data
    merged_data = pd.merge(predictions, odds_data, on='match_id', how='inner')
    
    if merged_data.empty:
        logger.warning("No matches found with both predictions and odds data.")
        return {}
    
    logger.info(f"Evaluating {len(merged_data)} bets")
    
    # Initialize results tracking
    results = []
    bankroll = initial_bankroll
    bets_placed = 0
    bets_won = 0
    profits = []
    
    # Evaluate each bet
    for _, bet in merged_data.iterrows():
        pred_prob = bet['pred_probability']
        odds_value = bet['odds']
        actual_win = bet['actual_result'] == 'win'
        
        # Calculate implied probability from odds
        implied_prob = 1 / odds_value
        
        # Calculate edge
        edge = pred_prob - implied_prob
        
        # Only bet if edge exceeds threshold
        if edge > threshold:
            # Determine stake amount
            if stake_method == 'kelly':
                stake = calculate_kelly_stake(pred_prob, odds_value, kelly_fraction) * bankroll
            else:  # flat staking
                stake = min(flat_stake, bankroll)  # Don't bet more than we have
            
            # Update tracking
            bets_placed += 1
            
            # Calculate result
            if actual_win:
                profit = stake * (odds_value - 1)
                bets_won += 1
            else:
                profit = -stake
            
            bankroll += profit
            
            # Record bet details
            results.append({
                'match_id': bet.get('match_id', ''),
                'predicted_prob': pred_prob,
                'odds': odds_value,
                'implied_prob': implied_prob,
                'edge': edge,
                'stake': stake,
                'result': 'win' if actual_win else 'loss',
                'profit': profit,
                'bankroll': bankroll
            })
            
            profits.append(profit)
    
    # Calculate evaluation metrics
    final_bankroll = bankroll
    roi = (final_bankroll - initial_bankroll) / initial_bankroll if bets_placed > 0 else 0
    win_rate = bets_won / bets_placed if bets_placed > 0 else 0
    
    # Create running profit/loss chart
    if results:
        results_df = pd.DataFrame(results)
        results_df['cumulative_profit'] = results_df['profit'].cumsum()
        results_df['cumulative_roi'] = results_df['cumulative_profit'] / initial_bankroll
    else:
        results_df = pd.DataFrame()
    
    # Calculate drawdown
    if not results_df.empty:
        results_df['peak'] = results_df['bankroll'].cummax()
        results_df['drawdown'] = (results_df['peak'] - results_df['bankroll']) / results_df['peak']
        max_drawdown = results_df['drawdown'].max()
    else:
        max_drawdown = 0.0
    
    # Summary metrics
    metrics = {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': final_bankroll,
        'roi': roi,
        'profit': final_bankroll - initial_bankroll,
        'bets_placed': bets_placed,
        'bets_won': bets_won,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'results_df': results_df
    }
    
    logger.info(f"Strategy evaluation complete. ROI: {roi:.2%}, Profit: ${final_bankroll - initial_bankroll:.2f}")
    
    return metrics

def plot_betting_results(metrics: Dict, output_path: Optional[str] = None):
    """
    Plot the results of a betting strategy evaluation.
    
    Args:
        metrics: Dictionary of metrics from evaluate_betting_strategy
        output_path: Optional path to save the plot
    """
    if not metrics or 'results_df' not in metrics or metrics['results_df'].empty:
        logger.warning("No betting results to plot.")
        return
    
    results_df = metrics['results_df']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot cumulative profit
    axes[0].plot(results_df.index, results_df['cumulative_profit'], 'b-')
    axes[0].set_title('Cumulative Profit/Loss')
    axes[0].set_ylabel('Profit ($)')
    axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0].grid(True)
    
    # Plot bankroll
    axes[1].plot(results_df.index, results_df['bankroll'], 'g-')
    axes[1].set_title('Bankroll Progression')
    axes[1].set_ylabel('Bankroll ($)')
    axes[1].axhline(y=metrics['initial_bankroll'], color='r', linestyle='-', alpha=0.3)
    axes[1].grid(True)
    
    # Plot drawdown
    axes[2].fill_between(results_df.index, 0, results_df['drawdown'], color='r', alpha=0.3)
    axes[2].set_title('Drawdown')
    axes[2].set_ylabel('Drawdown (%)')
    axes[2].set_xlabel('Bet Number')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Betting results plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_value_bets(predictions: pd.DataFrame, odds_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze value bets by comparing predicted probabilities with implied odds.
    
    Args:
        predictions: DataFrame with match predictions
        odds_data: DataFrame with odds data
        
    Returns:
        DataFrame with value bet analysis
    """
    # Merge predictions with odds data
    merged_data = pd.merge(predictions, odds_data, on='match_id', how='inner')
    
    if merged_data.empty:
        logger.warning("No matches found with both predictions and odds data.")
        return pd.DataFrame()
    
    # Calculate implied probability from odds
    merged_data['implied_prob'] = 1 / merged_data['odds']
    
    # Calculate edge
    merged_data['edge'] = merged_data['pred_probability'] - merged_data['implied_prob']
    
    # Sort by edge in descending order
    value_bets = merged_data.sort_values('edge', ascending=False)
    
    return value_bets

def simulate_betting_strategies(
    predictions: pd.DataFrame,
    odds_data: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    strategies: Optional[List[Dict]] = None
) -> Dict[str, Dict]:
    """
    Simulate multiple betting strategies and compare results.
    
    Args:
        predictions: DataFrame with match predictions
        odds_data: DataFrame with odds data
        initial_bankroll: Starting bankroll
        strategies: List of strategy parameter dictionaries
        
    Returns:
        Dictionary mapping strategy names to evaluation metrics
    """
    if strategies is None:
        # Default strategies to compare
        strategies = [
            {'name': 'Flat (1%)', 'stake_method': 'flat', 'flat_stake': initial_bankroll * 0.01, 'threshold': 0.0},
            {'name': 'Flat (5%)', 'stake_method': 'flat', 'flat_stake': initial_bankroll * 0.05, 'threshold': 0.0},
            {'name': 'Kelly (50%)', 'stake_method': 'kelly', 'kelly_fraction': 0.5, 'threshold': 0.0},
            {'name': 'Kelly (25%)', 'stake_method': 'kelly', 'kelly_fraction': 0.25, 'threshold': 0.0},
            {'name': 'Value (>5%)', 'stake_method': 'flat', 'flat_stake': initial_bankroll * 0.01, 'threshold': 0.05},
        ]
    
    # Simulate each strategy
    results = {}
    for strategy in strategies:
        name = strategy.pop('name')
        logger.info(f"Simulating strategy: {name}")
        
        metrics = evaluate_betting_strategy(
            predictions=predictions, 
            odds_data=odds_data, 
            initial_bankroll=initial_bankroll,
            **strategy
        )
        
        results[name] = metrics
    
    return results

def plot_strategy_comparison(
    strategy_results: Dict[str, Dict],
    output_path: Optional[str] = None
):
    """
    Plot a comparison of multiple betting strategies.
    
    Args:
        strategy_results: Dictionary mapping strategy names to evaluation metrics
        output_path: Optional path to save the plot
    """
    if not strategy_results:
        logger.warning("No strategy results to plot.")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot bankroll progression for each strategy
    for name, metrics in strategy_results.items():
        if 'results_df' not in metrics or metrics['results_df'].empty:
            continue
            
        results_df = metrics['results_df']
        ax1.plot(results_df.index, results_df['bankroll'], label=name)
    
    ax1.set_title('Bankroll Progression by Strategy')
    ax1.set_ylabel('Bankroll ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot ROI comparison
    strategy_names = []
    rois = []
    for name, metrics in strategy_results.items():
        strategy_names.append(name)
        rois.append(metrics.get('roi', 0) * 100)  # Convert to percentage
    
    ax2.bar(strategy_names, rois)
    ax2.set_title('ROI by Strategy')
    ax2.set_ylabel('ROI (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Strategy comparison plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
def main():
    """Example usage of the betting evaluation functions."""
    # This is a placeholder example
    logger.info("This module provides betting evaluation utilities. Import and use the functions in your code.")

if __name__ == "__main__":
    main() 