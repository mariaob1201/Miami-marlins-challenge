# Source https://www.pymc-labs.com/blog-posts/probabilistic-forecasting, https://juanitorduz.github.io/html/pyconco22_orduz.html#/references
# I want to reproduce this: https://minimizeregret.com/short-time-series-prior-knowledge
# to forecast short time series using bayesian transfer learning as a first approach for the Miami Marlins challenge

import warnings

warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# PyMC 5.x
import pymc as pm
import pytensor.tensor as at
import arviz as az


def load_and_prepare_data(file_path='1_Data/exit_velo_project_data.csv'):
    """
    Load and prepare the baseball data with proper error handling
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        df = df[df['level_abbr'] == 'mlb']  # <- change path

        # Check if required columns exist
        required_cols = ['batter_id', 'season', 'exit_velo', 'level_abbr']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print("Available columns:")
            for col in df.columns:
                print(f"  - {col}")
            return None

        # Filter for MLB data
        if 'level_abbr' in df.columns:
            df = df[df['level_abbr'] == 'mlb']
            print(f"After filtering for MLB: {df.shape}")

        # Ensure season is integer
        df["season"] = df["season"].astype(int)

        # Check data ranges
        print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
        print(f"Number of unique players: {df['batter_id'].nunique()}")
        print(f"Exit velocity range: {df['exit_velo'].min():.1f} - {df['exit_velo'].max():.1f}")

        return df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Creating sample data for demonstration...")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data for demonstration...")
        return create_sample_data()


def create_sample_data():
    """
    Create sample data that matches the expected format
    """
    np.random.seed(42)

    # Create sample data
    players = [f"player_{i}" for i in range(1, 51)]  # 50 players
    seasons = [2021, 2022, 2023]  # 3 seasons

    data = []

    for player in players:
        base_ev = 85 + np.random.normal(0, 5)  # Player baseline
        trend = np.random.normal(0, 1)  # Season-to-season trend

        for i, season in enumerate(seasons):
            # Not all players have data for all seasons
            if np.random.random() < 0.85:  # 85% chance of having data
                n_events = np.random.randint(50, 200)  # Variable sample sizes

                # Season effect
                season_effect = trend * i + np.random.normal(0, 1.5)
                season_mean = base_ev + season_effect

                # Generate individual events
                events = np.random.normal(season_mean, 3.5, n_events)

                for ev in events:
                    data.append({
                        'batter_id': player,
                        'season': season,
                        'exit_velo': max(ev, 60),  # Minimum realistic EV
                        'level_abbr': 'mlb'
                    })

    df = pd.DataFrame(data)
    print(f"Created sample data with {len(df)} events")
    return df


def prepare_model_data(df):
    """
    Prepare data for PyMC model with proper validation
    """
    print("\n=== PREPARING MODEL DATA ===")

    # Per-player-per-season summaries
    agg = (
        df.groupby(["batter_id", "season"])
        .agg(
            ev_mean=("exit_velo", "mean"),
            ev_sd=("exit_velo", "std"),
            n=("exit_velo", "size")
        )
        .reset_index()
    )

    print(f"Aggregated to {len(agg)} player-season combinations")

    # Filter players with sufficient data
    player_counts = agg.groupby('batter_id').size()
    players_with_enough_data = player_counts[player_counts >= 2].index
    agg = agg[agg['batter_id'].isin(players_with_enough_data)]

    print(f"Players with 2+ seasons: {len(players_with_enough_data)}")

    # Seasons present and define next season to forecast
    seasons_present = np.sort(agg["season"].unique())
    next_season = seasons_present.max() + 1
    all_seasons = np.r_[seasons_present, next_season]  # include future

    players = np.sort(agg["batter_id"].unique())
    n_players = len(players)
    T = len(all_seasons)

    print(f"Final dataset: {n_players} players, {T} seasons (including forecast)")
    print(f"Seasons: {seasons_present} + forecast for {next_season}")

    # Build dense matrices (player x season) with NaNs for missing cells
    y = np.full((n_players, T), np.nan, dtype=float)  # season means
    n_mat = np.zeros((n_players, T), dtype=int)  # event counts

    # Fill in observed data
    for _, row in agg.iterrows():
        player_idx = np.where(players == row['batter_id'])[0][0]
        season_idx = np.where(all_seasons == row['season'])[0][0]

        y[player_idx, season_idx] = row['ev_mean']
        n_mat[player_idx, season_idx] = row['n']

    # Create observation masks and indices
    obs_mask = ~np.isnan(y)
    i_idx, t_idx = np.where(obs_mask)
    y_obs = y[obs_mask].astype(float)
    sqrt_n = np.sqrt(np.clip(n_mat[obs_mask], 1, None))

    print(f"Total observations: {len(y_obs)}")

    # Standardize EV to improve sampling geometry
    ev_mean = np.nanmean(y)
    ev_std = np.nanstd(y)

    if ev_std == 0 or np.isnan(ev_std):
        ev_std = 1.0
        print("Warning: Setting ev_std to 1.0 due to zero or NaN variance")

    print(f"Standardization: mean={ev_mean:.2f}, std={ev_std:.2f}")

    y_std = (y - ev_mean) / ev_std
    y_obs_std = y_std[obs_mask]

    return {
        'y': y,
        'y_std': y_std,
        'y_obs_std': y_obs_std,
        'n_mat': n_mat,
        'obs_mask': obs_mask,
        'i_idx': i_idx,
        't_idx': t_idx,
        'sqrt_n': sqrt_n,
        'players': players,
        'all_seasons': all_seasons,
        'seasons_present': seasons_present,
        'next_season': next_season,
        'n_players': n_players,
        'T': T,
        'ev_mean': ev_mean,
        'ev_std': ev_std,
        'agg': agg
    }


def build_and_sample_model(data_dict):
    """
    Build and sample the PyMC model with proper error handling
    """
    print("\n=== BUILDING PYMC MODEL ===")

    # Extract data
    y_obs_std = data_dict['y_obs_std']
    i_idx = data_dict['i_idx']
    t_idx = data_dict['t_idx']
    sqrt_n = data_dict['sqrt_n']
    players = data_dict['players']
    all_seasons = data_dict['all_seasons']
    n_players = data_dict['n_players']
    T = data_dict['T']

    # Set up coordinates
    coords = {"player": players, "season": all_seasons}

    print(f"Model dimensions: {n_players} players × {T} seasons")
    print(f"Observations: {len(y_obs_std)}")

    with pm.Model(coords=coords) as model:

        # Global parameters
        mu_global = pm.Normal("mu_global", mu=0.0, sigma=1.0)

        # Between-player heterogeneity
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=1.0)
        mu_player = pm.Normal("mu_player", mu=mu_global, sigma=sigma_mu, dims="player")

        # Random walk components
        sigma_init = pm.HalfNormal("sigma_init", sigma=1.0)
        delta0 = pm.Normal("delta0", mu=0.0, sigma=sigma_init, dims="player")

        sigma_state = pm.HalfNormal("sigma_state", sigma=0.5)

        # Random walk increments
        eps = pm.Normal("eps", mu=0.0, sigma=1.0, dims=("player", "season"))

        # Set first season increment to 0
        eps_corrected = at.set_subtensor(eps[:, 0], 0.0)

        # Cumulative sum for random walk
        delta = pm.Deterministic(
            "delta",
            delta0[:, None] + at.cumsum(eps_corrected * sigma_state, axis=1),
            dims=("player", "season")
        )

        # Latent true performance (standardized)
        theta_std = pm.Deterministic(
            "theta_std",
            mu_player[:, None] + delta,
            dims=("player", "season")
        )

        # Observation model
        sigma_event_std = pm.HalfNormal("sigma_event_std", sigma=0.7)

        # Extract observations using fancy indexing
        mu_obs = theta_std[i_idx, t_idx]
        sigma_obs = sigma_event_std / sqrt_n

        # Likelihood
        y_like = pm.Normal("y_like", mu=mu_obs, sigma=sigma_obs, observed=y_obs_std)

        print("Model built successfully")

        # Sample with robust settings
        print("Starting MCMC sampling...")
        try:
            idata = pm.sample(
                tune=1500,
                draws=1500,
                chains=4,
                cores=4,
                target_accept=0.95,
                max_treedepth=12,
                random_seed=123,
                progressbar=True
            )
            print("✓ Sampling completed successfully")

        except Exception as e:
            print(f"Sampling failed with error: {e}")
            print("Trying with more conservative settings...")

            idata = pm.sample(
                tune=1000,
                draws=1000,
                chains=2,
                cores=2,
                target_accept=0.90,
                init="adapt_diag",
                random_seed=123,
                progressbar=True
            )
            print("✓ Sampling completed with conservative settings")

    return model, idata


def check_model_diagnostics(idata):
    """
    Check MCMC diagnostics
    """
    print("\n=== MODEL DIAGNOSTICS ===")

    # R-hat (should be close to 1.0)
    rhat_max = az.rhat(idata).to_array().max().item()
    print(f"Max R-hat: {rhat_max:.4f}")

    if rhat_max > 1.1:
        print("⚠️  Warning: High R-hat values indicate poor convergence")
    else:
        print("✓ R-hat values look good")

    # Effective sample size
    ess_bulk_min = az.ess(idata, method="bulk").to_array().min().item()
    ess_tail_min = az.ess(idata, method="tail").to_array().min().item()

    print(f"Min bulk ESS: {ess_bulk_min:.0f}")
    print(f"Min tail ESS: {ess_tail_min:.0f}")

    if ess_bulk_min < 100 or ess_tail_min < 100:
        print("⚠️  Warning: Low effective sample sizes")
    else:
        print("✓ Effective sample sizes look adequate")

    # Energy diagnostics
    try:
        energy_stats = az.bfmi(idata)
        print(f"BFMI: {energy_stats.mean().item():.3f}")

        if energy_stats.mean().item() < 0.2:
            print("⚠️  Warning: Low BFMI indicates potential sampling issues")
    except:
        print("Could not compute energy diagnostics")


def generate_forecasts(idata, data_dict):
    """
    Generate forecasts for next season
    """
    print("\n=== GENERATING FORECASTS ===")

    # Extract data
    players = data_dict['players']
    next_season = data_dict['next_season']
    ev_mean = data_dict['ev_mean']
    ev_std = data_dict['ev_std']
    n_mat = data_dict['n_mat']
    agg = data_dict['agg']

    # Get posterior samples for next season
    theta_post = idata.posterior["theta_std"]  # dims: chain, draw, player, season
    theta_next_std = theta_post.sel(season=next_season)  # chain, draw, player

    # Convert back to original units (mph)
    theta_next = (theta_next_std * ev_std) + ev_mean

    # Calculate credible intervals for latent true mean
    theta_next_q = theta_next.quantile([0.025, 0.5, 0.975], dim=("chain", "draw"))

    # Convert to DataFrame
    latent_summary = pd.DataFrame({
        'batter_id': players,
        'latent_lo': theta_next_q.sel(quantile=0.025).values,
        'latent_med': theta_next_q.sel(quantile=0.5).values,
        'latent_hi': theta_next_q.sel(quantile=0.975).values
    })

    # Estimate future sample sizes (for predictive intervals)
    past_counts = []
    for player in players:
        player_data = agg[agg['batter_id'] == player]
        if len(player_data) > 0:
            avg_n = player_data['n'].median()
        else:
            avg_n = 100  # Default
        past_counts.append(int(avg_n))

    assumed_n_future = np.array(past_counts)

    # Generate predictive samples for observed season averages
    print("Generating predictive samples...")

    theta_next_np = theta_next.values  # (chain, draw, player)
    sigma_event_std_samples = idata.posterior["sigma_event_std"].values  # (chain, draw)

    # Convert observation error back to original units
    sigma_event_samples = sigma_event_std_samples * ev_std

    # Predictive samples
    n_total_samples = theta_next_np.shape[0] * theta_next_np.shape[1]
    theta_flat = theta_next_np.reshape(n_total_samples, -1)  # (total_samples, player)
    sigma_flat = sigma_event_samples.flatten()[:n_total_samples]  # (total_samples,)

    # Observation standard deviation for each player
    obs_std_matrix = sigma_flat[:, None] / np.sqrt(assumed_n_future)[None, :]

    # Generate predictive samples
    rng = np.random.default_rng(123)
    y_pred_samples = rng.normal(theta_flat, obs_std_matrix)

    # Calculate quantiles
    obs_quantiles = np.quantile(y_pred_samples, [0.025, 0.5, 0.975], axis=0)

    obs_summary = pd.DataFrame({
        'batter_id': players,
        'pred_obs_lo': obs_quantiles[0],
        'pred_obs_med': obs_quantiles[1],
        'pred_obs_hi': obs_quantiles[2],
        'assumed_n_future': assumed_n_future
    })

    # Combine results
    forecast = latent_summary.merge(obs_summary, on='batter_id')
    forecast = forecast.sort_values('latent_med', ascending=False).reset_index(drop=True)

    # Add some useful metrics
    forecast['latent_width'] = forecast['latent_hi'] - forecast['latent_lo']
    forecast['pred_width'] = forecast['pred_obs_hi'] - forecast['pred_obs_lo']
    forecast['uncertainty_ratio'] = forecast['pred_width'] / forecast['latent_width']

    return forecast, theta_next


def main():
    """
    Main execution function
    """
    print("🏀 BAYESIAN TRANSFER LEARNING FOR BASEBALL EXIT VELOCITY")
    print("=" * 60)

    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Prepare for modeling
    data_dict = prepare_model_data(df)

    # Step 3: Build and sample model
    model, idata = build_and_sample_model(data_dict)

    # Step 4: Check diagnostics
    check_model_diagnostics(idata)

    # Step 5: Generate forecasts
    forecast, theta_next = generate_forecasts(idata, data_dict)

    # Step 6: Display results
    print(f"\n=== FORECAST RESULTS ===")
    print(f"Next season: {data_dict['next_season']}")
    print(f"Number of players: {len(forecast)}")

    # Show top performers
    print(f"\nTop 10 predicted performers for {data_dict['next_season']}:")
    top_10 = forecast.head(10)

    for i, row in top_10.iterrows():
        print(f"{i + 1:2d}. {row['batter_id']:<12} "
              f"{row['latent_med']:5.1f} mph "
              f"[{row['latent_lo']:5.1f}, {row['latent_hi']:5.1f}] "
              f"(width: {row['latent_width']:4.1f})")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean predicted EV: {forecast['latent_med'].mean():.1f} mph")
    print(f"  Std of predictions: {forecast['latent_med'].std():.1f} mph")
    print(f"  Avg credible interval width: {forecast['latent_width'].mean():.1f} mph")
    print(f"  Avg predictive interval width: {forecast['pred_width'].mean():.1f} mph")

    # Step 7: Save results
    try:
        forecast.to_csv("predictions_next_season_by_player.csv", index=False)
        print(f"\n✓ Saved predictions to: predictions_next_season_by_player.csv")

        # Save posterior samples if desired
        theta_next_df = pd.DataFrame(
            theta_next.values.reshape(-1, len(data_dict['players'])),
            columns=data_dict['players']
        )
        theta_next_df.to_csv("posterior_draws_theta_next_latent.csv", index=False)
        print(f"✓ Saved posterior draws to: posterior_draws_theta_next_latent.csv")

    except Exception as e:
        print(f"Error saving files: {e}")

    print(f"\n🎉 Analysis complete!")

    return model, idata, forecast, data_dict


if __name__ == "__main__":
    model, idata, forecast, data_dict = main()