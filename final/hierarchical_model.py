import pandas as pd
import numpy as np
import warnings

import pymc as pm
import arviz as az

RANDOM_SEED = 40

def build_hierarchical_model(df):
    """Build the hierarchical Bayesian model"""
    print("Building hierarchical Bayesian model...")

    df_model = df.copy()
    # --- BEFORE mapping, normalize the text ---
    df_model['level_abbr'] = (
        df_model['level_abbr']
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            'TRIPLE-A': 'AAA', 'AAA BALL': 'AAA', 'aaa':'AAA',
            'DOUBLE-A': 'AA','aa': 'AA',
            'MAJOR LEAGUE': 'MLB', 'ML': 'MLB', 'mlb': 'MLB'
        })
    )

    level_map = {'MLB': 0, 'AAA': 1, 'AA': 2}
    df_model['level_idx'] = df_model['level_abbr'].map(level_map)

    # Better unknown-level check
    if df_model['level_idx'].isna().any():
        unknown = np.sort(df_model.loc[df_model['level_idx'].isna(), 'level_abbr'].unique())
        raise ValueError(
            f"Unknown level_abbr values: {unknown.tolist()} "
            f"(after normalization). Expected one of {list(level_map)}"
        )

    # Level mapping & validation
    level_map = {'MLB': 0, 'AAA': 1, 'AA': 2}
    df_model['level_idx'] = df_model['level_abbr'].map(level_map)
    if df_model['level_idx'].isna().any():
        unknown = df_model.loc[df_model['level_idx'].isna(), 'level_abbr'].unique()
        raise ValueError(f"Unknown level_abbr values: {unknown}. Expected one of {list(level_map)}")

    # Center age (and ensure numeric)
    df_model['age'] = pd.to_numeric(df_model['age'], errors='coerce')
    mean_age = df_model['age'].mean()
    df_model['age_centered'] = df_model['age'] - mean_age

    # Ensure handedness_matchup exists (e.g., 1=same-handed, 0=opposite)
    if 'handedness_matchup' not in df_model:
        warnings.warn("Column 'handedness_matchup' not found; filling with zeros.")
        df_model['handedness_matchup'] = 0

    # Create indices
    batter_ids = pd.Categorical(df_model['batter_id'])
    pitcher_ids = pd.Categorical(df_model['pitcher_id'])

    n_players = len(batter_ids.categories)
    n_pitchers = len(pitcher_ids.categories)

    print(f"Model dimensions: {n_players} players, {n_pitchers} pitchers, {len(df_model)} observations")

    with pm.Model() as model:
        # Hyperprior on global mean EV
        mu_global = pm.Normal('mu_global', mu=90, sigma=10)

        # Level adjustments (MLB=0 reference)
        alpha_level = pm.Normal('alpha_level',
                                mu=np.array([0.0, -2.5, -4.5]),
                                sigma=1.0,
                                shape=3)

        # Age effects (centered age; quadratic prefers small prior scale)
        beta_age_linear = pm.Normal('beta_age_linear', mu=0.2, sigma=0.1)
        beta_age_quad   = pm.Normal('beta_age_quad',   mu=-0.01, sigma=0.005)

        # Variance components
        sigma_batter = pm.HalfNormal('sigma_batter', sigma=10)
        sigma_pitcher = pm.HalfNormal('sigma_pitcher', sigma=2)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)

        # Random effects
        theta_batter = pm.Normal('theta_batter', mu=0, sigma=sigma_batter, shape=n_players)
        gamma_pitcher = pm.Normal('gamma_pitcher', mu=0, sigma=sigma_pitcher, shape=n_pitchers)

        # Additional effect
        delta_handedness = pm.Normal('delta_handedness', mu=0, sigma=1)

        # Linear predictor
        age_c = df_model['age_centered'].to_numpy()
        mu = (
            mu_global
            + theta_batter[batter_ids.codes]
            + alpha_level[df_model['level_idx'].to_numpy()]
            + beta_age_linear * age_c
            + beta_age_quad * (age_c ** 2)
            + gamma_pitcher[pitcher_ids.codes]
            + delta_handedness * df_model['handedness_matchup'].to_numpy()
        )

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs,
                          observed=df_model['exit_velo'].to_numpy())

    return model, df_model, batter_ids, pitcher_ids


def sample_model(model, draws=1000, tune=1000, chains=2):
    """Sample from the model and run diagnostics"""
    print(f"Sampling model: {draws} draws, {tune} tune, {chains} chains...")

    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          random_seed=RANDOM_SEED, progressbar=True)

        prior_pred = pm.sample_prior_predictive(samples=100, random_seed=RANDOM_SEED)
        post_pred  = pm.sample_posterior_predictive(trace, random_seed=RANDOM_SEED)

    return trace, prior_pred, post_pred


def analyze_model_results(trace, model, text_only=True):
    """
    Analyze model results and diagnostics.
    Set text_only=True to avoid ArviZ rich HTML assets (workaround for your error).
    """
    print("Model Convergence Diagnostics")
    print("=" * 35)

    # If ArviZ HTML assets are broken, keep it plain
    if text_only:
        # az.summary returns a DataFrame; printing .to_string() avoids HTML repr
        summary = az.summary(trace, round_to=3)
        print(summary.to_string())
    else:
        summary = az.summary(trace, round_to=3)
        print(summary)

    # Key parameters
    key_params = ['mu_global', 'beta_age_linear', 'beta_age_quad',
                  'sigma_batter', 'sigma_pitcher', 'sigma_obs']
    print("\nKey Parameter Estimates:")
    print("-" * 25)
    for param in key_params:
        if param in summary.index:
            row = summary.loc[param]
            print(f"{param:<20}: {row['mean']:7.3f} [{row['hdi_3%']:7.3f}, {row['hdi_97%']:7.3f}]")

    # Level adjustments (vector params)
    alpha_rows = summary[summary.index.str.contains(r'^alpha_level(\[|\b)')]
    if not alpha_rows.empty:
        levels = ['MLB', 'AAA', 'AA']
        print("\nLevel Adjustments:")
        for i, level in enumerate(levels):
            key = f'alpha_level[{i}]' if f'alpha_level[{i}]' in alpha_rows.index else 'alpha_level'
            if key in alpha_rows.index:
                row = alpha_rows.loc[key]
                print(f"  {level:<10}: {row['mean']:7.3f} [{row['hdi_3%']:7.3f}, {row['hdi_97%']:7.3f}]")

    # R-hat diagnostics
    if 'r_hat' in summary.columns:
        rhat_issues = summary[summary['r_hat'] > 1.1]
        if len(rhat_issues) > 0:
            print(f"\nWarning: {len(rhat_issues)} parameters have R-hat > 1.1")
        else:
            print("\n✓ All parameters have R-hat < 1.1 (good convergence)")

    return summary
