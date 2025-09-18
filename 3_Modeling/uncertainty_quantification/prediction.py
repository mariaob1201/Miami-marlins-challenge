# https://towardsdatascience.com/using-bayesian-modeling-to-predict-the-champions-league-8ebb069006ba/
# https://academic.oup.com/jrsssa/article/187/2/513/7512935
# https://github.com/alan-turing-institute/pymc3
# https://github.com/marcopeix/TimeSeriesForecastingInPython/blob/master/CH04/CH04.ipynb
# start with arima model first

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
import os


def find_data_file(filename):
    """Helper function to find data file in various possible locations"""
    possible_paths = [
        filename,
        f'data/{filename}',
        f'1_Data/{filename}',
        f'../data/{filename}',
        f'../../data/{filename}'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def prepare_model_data(df):
    """Prepare data for PyMC model with proper encoding"""
    df = df.copy()

    # Handle missing values
    df = df.dropna(subset=['exit_velo', 'level_abbr', 'batter_id', 'age'])

    # Create level encoding
    level_mapping = {'MLB': 0, 'AAA': 1, 'AA': 2}
    df['level_idx'] = df['level_abbr'].map(level_mapping)

    # Center age for numerical stability
    age_mean = df['age'].mean()
    df['age_centered'] = df['age'] - age_mean

    # Create player indices
    unique_players = df['batter_id'].unique()
    player_map = {pid: idx for idx, pid in enumerate(unique_players)}
    df['player_idx'] = df['batter_id'].map(player_map)

    # Convert to numpy arrays with proper dtypes
    model_data = {
        'exit_velo': df['exit_velo'].values.astype(np.float64),
        'level_idx': df['level_idx'].values.astype(int),
        'age_centered': df['age_centered'].values.astype(np.float64),
        'player_idx': df['player_idx'].values.astype(int),
        'n_players': len(unique_players),
        'n_obs': len(df),
        'age_mean': age_mean,
        'player_map': player_map,
        'level_mapping': level_mapping
    }

    return model_data, df


def build_simple_hierarchical_model(data):
    """Build a simple but robust hierarchical model"""

    with pm.Model() as model:
        # Global intercept
        alpha = pm.Normal('alpha', mu=89, sigma=5)

        # Level effects (MLB is reference, so AAA and AA have adjustments)
        # Use separate parameters for each level for cleaner indexing
        level_mlb = pm.Normal('level_mlb', mu=0, sigma=0.1)  # Reference level
        level_aaa = pm.Normal('level_aaa', mu=-1.5, sigma=1)  # AAA adjustment
        level_aa = pm.Normal('level_aa', mu=-3.0, sigma=1)  # AA adjustment

        # Stack into array for indexing
        level_effects = pm.math.stack([level_mlb, level_aaa, level_aa])

        # Age effects (linear and quadratic)
        age_coef = pm.Normal('age_coef', mu=0, sigma=0.5)
        age_quad_coef = pm.Normal('age_quad_coef', mu=-0.05, sigma=0.1)

        # Player random effects
        sigma_player = pm.HalfNormal('sigma_player', sigma=3)
        player_effects = pm.Normal('player_effects',
                                   mu=0,
                                   sigma=sigma_player,
                                   shape=data['n_players'])

        # Observation noise
        sigma = pm.HalfNormal('sigma', sigma=2)

        # Build mean function
        age_effect = (age_coef * data['age_centered'] +
                      age_quad_coef * data['age_centered'] ** 2)

        # Use advanced indexing that PyMC handles better
        level_effect = level_effects[data['level_idx']]
        player_effect = player_effects[data['player_idx']]

        mu = alpha + level_effect + age_effect + player_effect

        # Likelihood
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data['exit_velo'])

    return model


def fit_and_diagnose_model(model, draws=1000, tune=1000, chains=2):
    """Fit model and run basic diagnostics"""

    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, return_inferencedata=True)

        print("=== Model Diagnostics ===")

        # Use summary table (cleaner than raw Datasets)
        summary = az.summary(trace)

        max_rhat = summary["r_hat"].max()
        min_ess = summary["ess_bulk"].min()

        print(f"Max R-hat: {max_rhat:.4f}")
        if max_rhat > 1.01:
            print("WARNING: Some parameters may not have converged (R-hat > 1.01)")
        else:
            print("✓ All parameters appear to have converged")

        print(f"Min effective sample size: {min_ess:.0f}")
        if min_ess < 100:
            print("WARNING: Low effective sample size for some parameters")
        else:
            print("✓ Adequate effective sample sizes")

        print("\n=== Parameter Summary ===")
        print(summary.loc[["alpha", "level_aaa", "level_aa", "age_coef", "sigma_player", "sigma"]])

    return trace


def make_predictions(trace, model_data, new_data):
    """Generate predictions for new players/scenarios"""
    predictions = []

    # Extract posterior samples
    alpha_samples = trace.posterior['alpha'].values.flatten()
    level_aaa_samples = trace.posterior['level_aaa'].values.flatten()
    level_aa_samples = trace.posterior['level_aa'].values.flatten()
    age_coef_samples = trace.posterior['age_coef'].values.flatten()
    age_quad_samples = trace.posterior['age_quad_coef'].values.flatten()
    sigma_player_samples = trace.posterior['sigma_player'].values.flatten()

    for _, row in new_data.iterrows():
        # Get level effect
        if row['level_abbr'] == 'MLB':
            level_effect_samples = np.zeros_like(alpha_samples)
        elif row['level_abbr'] == 'AAA':
            level_effect_samples = level_aaa_samples
        else:  # AA
            level_effect_samples = level_aa_samples

        # Age effect
        age_centered = row['age'] - model_data['age_mean']
        age_effect_samples = (age_coef_samples * age_centered +
                              age_quad_samples * age_centered ** 2)

        # For new players, sample from population distribution
        # For existing players, would use their specific effect
        player_effect_samples = np.random.normal(0, sigma_player_samples)

        # Predicted exit velocity
        pred_samples = (alpha_samples + level_effect_samples +
                        age_effect_samples + player_effect_samples)

        predictions.append({
            'batter_id': row['batter_id'],
            'predicted_exit_velo': np.mean(pred_samples),
            'pred_std': np.std(pred_samples),
            'pred_lower': np.percentile(pred_samples, 2.5),
            'pred_upper': np.percentile(pred_samples, 97.5)
        })

    return pd.DataFrame(predictions)

def build_prediction_dataset(df_clean, target_season=2024):
    """Build new dataset of players with projected age/level for prediction"""
    # Get last known season per player
    last_records = df_clean.groupby("batter_id").tail(1).copy()

    # Increment season to prediction year
    last_records["season"] = target_season
    last_records["age"] = last_records["age"] + (target_season - last_records["season"].min())

    return last_records[["batter_id", "age", "level_abbr"]]

def main():
    """Main execution function"""

    print("=== PyMC Bayesian Exit Velocity Model ===\n")

    # Find and load data
    data_file = find_data_file('1_Data/exit_velo_project_data.csv')
    if data_file is None:
        print("ERROR: Could not find data file 'exit_velo_project_data.csv'")
        print(f"Current directory: {os.getcwd()}")
        print("Please ensure the data file is in the correct location")
        return

    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    print(f"Original data shape: {df.shape}")
    print(f"Levels in data: {df['level_abbr'].value_counts().to_dict()}")

    # For faster testing, use a subset
    if len(df) > 5000:
        print("Using random subset of 5000 observations for faster testing...")
        df = df.sample(n=5000, random_state=42)

    # Prepare data
    print("\nPreparing data for modeling...")
    model_data, df_clean = prepare_model_data(df)
    print(f"Clean data shape: {model_data['n_obs']} observations, {model_data['n_players']} players")

    # Build model
    print("\nBuilding hierarchical model...")
    model = build_simple_hierarchical_model(model_data)

    # Fit model
    print("\nFitting model...")
    try:
        trace = fit_and_diagnose_model(model, draws=500, tune=500, chains=2)
        print("\n✓ Model fitted successfully!")
    except Exception as e:
        print(e)
    # --- Build prediction dataset for 2024 ---
    new_data = build_prediction_dataset(df_clean, target_season=2024)

    # --- Make predictions ---
    preds_2024 = make_predictions(trace, model_data, new_data)

    # --- Save to CSV ---
    output_file = "predictions_exit_velo_2024.csv"
    preds_2024.to_csv(output_file, index=False)
    print(f"\n✓ Saved predictions to {output_file}")
    print(preds_2024.head())


if __name__ == "__main__":
    main()


