# AFL Betting Model Development

This directory contains the code and resources for developing predictive models for AFL betting markets. We're using an LLM-based approach for interactive data analysis instead of traditional Jupyter notebooks.

## Project Structure

- `data_exploration/`: Scripts for LLM-assisted data exploration
- `feature_engineering/`: Code for creating and transforming features
- `models/`: Model implementation and evaluation
- `config/`: Configuration files for models and data processing
- `utils/`: Utility functions for data handling and processing

## Development Approach

Rather than using traditional Jupyter notebooks, we'll use a combination of Python scripts and LLM interactions. This allows us to:

1. Document our thinking process
2. Explore data through guided conversations
3. Generate code that can be run and refined iteratively
4. Create a more reproducible workflow

## Next Steps & Guided Prompts

Below are the next steps in our development process, along with prompt templates to guide the LLM-based analysis. Use these prompts when interacting with the LLM to guide the exploration and modeling process.

### 1. Data Loading and Initial Exploration

**Prompt:**
```
Create a Python script to load and display summary statistics for the AFL match data, player data, and odds data. Focus on:
1. Data shape and completeness (missing values)
2. Time range covered by each dataset
3. Key variables and their distributions
4. Potential data quality issues

After the basic statistics, create visualizations for:
1. Match score distributions over time
2. Player performance trends
3. Odds distribution and potential biases

Show me the code and explain what insights we should look for.
```

### 2. Feature Engineering for Match Prediction

**Prompt:**
```
Develop a feature engineering pipeline for match prediction. Based on the AFL data, create features that capture:

1. Team performance metrics:
   - Recent form (last N matches)
   - Home/away performance differences
   - Scoring patterns by quarter
   - Defensive strength

2. Player-based features:
   - Availability of key players
   - Team strength based on player statistics
   - Player form and impact metrics

3. Contextual features:
   - Venue effects
   - Round of season
   - Days since last match

Provide the code as a Python module with clear documentation.
```

### 3. Baseline Model Development

**Prompt:**
```
Create a baseline prediction model for AFL match outcomes using the following approach:

1. Target variable: Match winner (binary classification)
2. Features: Use the key features identified in our feature engineering
3. Model: Implement a gradient boosting classifier (XGBoost or LightGBM)
4. Evaluation: Use appropriate metrics for classification (accuracy, ROC-AUC, log loss)
5. Validation: Implement time-based cross-validation

Include code for:
1. Model training
2. Hyperparameter tuning
3. Feature importance analysis
4. Performance evaluation

Also provide guidance on interpreting the results and identifying potential improvements.
```

### 4. Odds-Based Model Evaluation

**Prompt:**
```
Develop a framework to evaluate our model's performance in the context of betting markets:

1. Calculate model accuracy compared to bookmaker accuracy
2. Implement Kelly criterion for optimal bet sizing
3. Simulate betting strategies using our model predictions
4. Calculate ROI, profit/loss, and drawdown metrics

Show the code to:
1. Integrate our model predictions with historical odds data
2. Evaluate different betting thresholds
3. Visualize performance metrics over time
4. Identify specific conditions where our model outperforms the market
```

### 5. Advanced Model Development

**Prompt:**
```
Based on our baseline model performance, develop more advanced modeling approaches:

1. Ensemble method combining multiple prediction models
2. Time-aware features that capture seasonality and trends
3. Player-interaction features using network/graph approaches
4. Specialized models for different betting markets (match winner, over/under, etc.)

Implement at least two advanced modeling techniques and compare their performance to our baseline.

Also suggest ways to:
1. Reduce overfitting
2. Improve model stability over time
3. Handle class imbalance if present
4. Incorporate uncertainty in predictions
```

### 6. Model Deployment Strategy

**Prompt:**
```
Design a deployment strategy for our AFL betting model:

1. Create a pipeline for regular model updates
2. Develop an API for generating predictions
3. Implement monitoring for model performance and drift
4. Design a user interface for displaying predictions and recommended bets

Provide:
1. Code structure for deployment
2. Required infrastructure components
3. Schedule for model retraining
4. Monitoring metrics to track
```

## Working with This Repository

To work with this repository effectively:

1. Run the data loading and exploration scripts first
2. Review the exploration results and refine the feature engineering
3. Train baseline models before moving to advanced approaches
4. Always validate models on recent data not seen during training
5. Document your findings and insights for each step

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run initial data exploration
python -m model.data_exploration.explore_data

# Generate features
python -m model.feature_engineering.generate_features

# Train baseline model
python -m model.models.baseline_model
``` 