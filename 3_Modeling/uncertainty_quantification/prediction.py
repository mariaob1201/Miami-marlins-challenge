# https://towardsdatascience.com/using-bayesian-modeling-to-predict-the-champions-league-8ebb069006ba/
# https://academic.oup.com/jrsssa/article/187/2/513/7512935
# https://github.com/alan-turing-institute/pymc3
# https://github.com/marcopeix/TimeSeriesForecastingInPython/blob/master/CH04/CH04.ipynb
# start with arima model first
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


# Load and prepare data
def load_and_prep_data(filepath):
    """Load and preprocess the exit velocity data"""
    df = pd.read_csv(filepath)

    # Create level hierarchy mapping - handle league class
    level_map = {
        'mlb': 0,
        'aaa': 1,
        'aa': 2
    }
    df['level_numeric'] = df['level_abbr'].map(level_map)
    df['level_numeric'] = df['level_numeric'].astype(int)

    # Encode categorical variables and store mappings
    df['batter_idx'], batter_mapping = pd.factorize(df['batter_id'])
    df['pitcher_idx'], pitcher_mapping = pd.factorize(df['pitcher_id'])
    df['hit_type_idx'], hit_type_mapping = pd.factorize(df['hit_type'])
    df['pitch_group_idx'], pitch_group_mapping = pd.factorize(df['pitch_group'])

    # Center age around mean
    df['age_centered'] = df['age'] - df['age'].mean()

    # Create handedness matchup (0=same, 1=opposite)
    df['handedness_matchup'] = (df['batter_hand'] != df['pitcher_hand']).astype(int)

    # Store all mappings in a dictionary for easy access
    mappings = {
        'batter_mapping': batter_mapping,
        'pitcher_mapping': pitcher_mapping,
        'hit_type_mapping': hit_type_mapping,
        'pitch_group_mapping': pitch_group_mapping
    }

    return df, mappings


def build_exit_velocity_model(df):
    """
    Hierarchical model for exit velocity

    Key features:
    - Hierarchical structure by batter with partial pooling
    - Level equivalency adjustments (MLB, AAA, AA)
    - Age effects (quadratic to capture peak/decline)
    - Competition quality adjustments
    - Time trend effects
    - Uncertainty quantification
    """

    n_batters = df['batter_idx'].nunique()
    n_pitchers = df['pitcher_idx'].nunique()
    #n_seasons = df['season'].nunique()
    n_hit_types = df['hit_type_idx'].nunique()
    n_pitch_groups = df['pitch_group_idx'].nunique()

    with pm.Model() as model:
        # ============= HYPERPRIORS =============

        # Global intercept
        mu_global = pm.Normal('mu_global', mu=90, sigma=10)  # Average exit velo ~90 mph

        # Batter talent distribution hyperparameters
        mu_batter = pm.Normal('mu_batter', mu=0, sigma=5)
        sigma_batter = pm.HalfNormal('sigma_batter', sigma=10)

        # ============= LEVEL EQUIVALENCY =============

        # Level adjustments (relative to MLB)
        # AAA typically ~2-3 mph lower, AA ~4-5 mph lower
        level_adjustment = pm.Normal('level_adjustment',
                                     mu=[0, -2.5, -4.5],  # MLB, AAA, AA
                                     sigma=1,
                                     shape=3)

        # ============= BATTER EFFECTS =============

        # Individual batter talent (hierarchical)
        batter_talent = pm.Normal('batter_talent',
                                  mu=mu_batter,
                                  sigma=sigma_batter,
                                  shape=n_batters)

        # Age effects (quadratic to capture peak/decline)
        age_linear = pm.Normal('age_linear', mu=0.2, sigma=0.1)
        age_quadratic = pm.Normal('age_quadratic', mu=-0.01, sigma=0.005)

        # ============= COMPETITION ADJUSTMENTS =============

        # Pitcher quality effect (hierarchical)
        sigma_pitcher = pm.HalfNormal('sigma_pitcher', sigma=2)
        pitcher_effect = pm.Normal('pitcher_effect',
                                   mu=0,
                                   sigma=sigma_pitcher,
                                   shape=n_pitchers)

        # Handedness matchup effect
        handedness_effect = pm.Normal('handedness_effect', mu=0, sigma=1)

        # ============= SITUATIONAL EFFECTS =============

        # Hit type effects (ground ball vs fly ball vs line drive, etc.)
        hit_type_effect = pm.Normal('hit_type_effect',
                                    mu=0,
                                    sigma=2,
                                    shape=n_hit_types)

        # Pitch type effects
        pitch_group_effect = pm.Normal('pitch_group_effect',
                                       mu=0,
                                       sigma=1,
                                       shape=n_pitch_groups)

        # ============= TIME TREND =============

        # Linear time trend to capture evolution of game
        time_trend = pm.Normal('time_trend', mu=0, sigma=0.5)
        season_centered = df['season'] - df['season'].mean()

        # ============= MODEL STRUCTURE =============

        # Expected exit velocity
        mu = (mu_global +
              batter_talent[df['batter_idx'].values] +
              level_adjustment[df['level_numeric'].values] +
              age_linear * df['age_centered'].values +
              age_quadratic * (df['age_centered'].values ** 2) +
              pitcher_effect[df['pitcher_idx'].values] +
              handedness_effect * df['handedness_matchup'].values +
              hit_type_effect[df['hit_type_idx'].values] +
              pitch_group_effect[df['pitch_group_idx'].values] +
              time_trend * season_centered.values)

        # Observation noise
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)

        # Likelihood
        exit_velo_obs = pm.Normal('exit_velo_obs',
                                  mu=mu,
                                  sigma=sigma_obs,
                                  observed=df['exit_velo'].values)

        # ============= DERIVED QUANTITIES =============

        # MLB-equivalent talent for each batter
        mlb_equivalent_talent = pm.Deterministic('mlb_equivalent_talent',
                                                 batter_talent + mu_global)

        # 2024 projections (assuming average age, MLB level, neutral matchups)
        proj_2024 = pm.Deterministic('proj_2024',
                                     mu_global +
                                     batter_talent +
                                     time_trend * (2024 - df['season'].mean()))

    return model


def run_inference(model, df, draws=2000, tune=2000, chains=4):
    """Run MCMC sampling"""
    with model:
        # Sample
        trace = pm.sample(draws=draws,
                          tune=tune,
                          chains=chains,
                          cores=4,
                          return_inferencedata=True,
                          random_seed=42)

        # Prior predictive checks
        prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=42)

        # Posterior predictive checks
        post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

    return trace, prior_pred, post_pred


def analyze_results(trace, df, mappings):
    """Analyze and interpret results"""

    # Summary statistics
    summary = az.summary(trace, var_names=['mu_global', 'level_adjustment',
                                           'age_linear', 'age_quadratic',
                                           'sigma_batter', 'sigma_obs'])
    print("Model Summary:")
    print(summary)

    # Extract 2024 projections with uncertainty
    proj_2024 = trace.posterior['proj_2024'].values
    proj_mean = np.mean(proj_2024, axis=(0, 1))
    proj_std = np.std(proj_2024, axis=(0, 1))
    proj_q05 = np.percentile(proj_2024, 5, axis=(0, 1))
    proj_q95 = np.percentile(proj_2024, 95, axis=(0, 1))

    # Create projection dataframe
    unique_batters = df.groupby('batter_idx')['batter_id'].first().sort_index()
    batter_mapping = mappings['batter_mapping']
    batter_names = batter_mapping[unique_batters.index]

    projections = pd.DataFrame({
        'batter_id': batter_names,
        'proj_exit_velo_2024': proj_mean,
        'proj_std': proj_std,
        'proj_q05': proj_q05,
        'proj_q95': proj_q95,
        'confidence_interval_width': proj_q95 - proj_q05
    })

    # Add historical performance for comparison
    historical_avg = df.groupby('batter_id')['exit_velo'].agg(['mean', 'std', 'count'])
    projections = projections.merge(historical_avg, left_on='batter_id', right_index=True, how='left')
    projections.rename(columns={'mean': 'historical_avg', 'std': 'historical_std'}, inplace=True)

    # Calculate shrinkage (how much we pull toward group mean)
    projections['shrinkage'] = 1 - (projections['proj_std'] ** 2 /
                                    (projections['proj_std'] ** 2 + projections['historical_std'] ** 2 / projections[
                                        'count']))

    return projections, summary


def plot_diagnostics(trace, post_pred, df):
    """Create diagnostic plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Trace plots for key parameters
    az.plot_trace(trace, var_names=['mu_global', 'sigma_batter'], axes=axes[0])

    # 2. Posterior predictive check
    axes[1, 0].hist(df['exit_velo'], bins=50, alpha=0.7, density=True, label='Observed')
    post_pred_samples = post_pred.posterior_predictive['exit_velo_obs'].values
    for i in range(min(100, post_pred_samples.shape[1])):
        axes[1, 0].hist(post_pred_samples[0, i, :], bins=50, alpha=0.01,
                        density=True, color='red')
    axes[1, 0].set_xlabel('Exit Velocity (mph)')
    axes[1, 0].set_title('Posterior Predictive Check')
    axes[1, 0].legend()

    # 3. Level equivalency
    level_adj = trace.posterior['level_adjustment'].values
    level_names = ['MLB', 'AAA', 'AA']
    axes[1, 1].violinplot([level_adj[:, :, i].flatten() for i in range(3)])
    axes[1, 1].set_xticks(range(1, 4))
    axes[1, 1].set_xticklabels(level_names)
    axes[1, 1].set_ylabel('Exit Velocity Adjustment (mph)')
    axes[1, 1].set_title('Level Equivalency Adjustments')

    plt.tight_layout()
    plt.show()


# Example usage
def main():
    # Load data
    df, mappings = load_and_prep_data('1_Data/exit_velo_project_data.csv')

    print(f"Loaded {len(df)} observations")
    print(f"Unique batters: {df['batter_idx'].nunique()}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Levels: {df['level_abbr'].value_counts()}")

    # Build model
    model = build_exit_velocity_model(df)

    # Run inference (start with fewer draws for testing)
    trace, prior_pred, post_pred = run_inference(model, df, draws=1000, tune=1000)

    # Analyze results
    projections, summary = analyze_results(trace, df, mappings)

    # Save predictions to CSV
    projections.to_csv('3_Modeling/uncertainty_quantification/outputs/predictions_exit_velo_2024.csv',
                       index=False)

    # If you also want to save the full posterior samples for uncertainty analysis
    import pickle
    projection_samples = {
        'proj_2024_samples': trace.posterior['proj_2024'].values,  # Full posterior samples
        'batter_ids': mappings['batter_mapping']  # Use the batter mapping directly
    }
    with open('3_Modeling/uncertainty_quantification/outputs/projection_samples.pkl', 'wb') as f:
        pickle.dump(projection_samples, f)

    # Show top/bottom performers
    print("\nTop 10 Projected Exit Velocities for 2024:")
    print(projections.nlargest(10, 'proj_exit_velo_2024')[['batter_id', 'proj_exit_velo_2024', 'proj_q05', 'proj_q95']])

    print("\nMost Uncertain Projections (widest confidence intervals):")
    print(projections.nlargest(10, 'confidence_interval_width')[
              ['batter_id', 'proj_exit_velo_2024', 'confidence_interval_width', 'count']])

    # Diagnostic plots
    plot_diagnostics(trace, post_pred, df)

    return model, trace, projections


if __name__ == "__main__":
    model, trace, projections = main()