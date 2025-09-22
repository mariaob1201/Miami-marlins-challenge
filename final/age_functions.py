import numpy as np
import pandas as pd
import statsmodels.formula.api as smf



def analyze_age_effects(df, min_bbe=5, age_min=18, age_max=45):
    """
    Analyze age effects to justify a quadratic age term on exit velocity.

    Returns
    -------
    fit_all : statsmodels RegressionResultsWrapper
        WLS fit with cluster-robust SEs by batter.
    peak_age_overall : float or np.nan
        Estimated peak age (vertex) in years if concave; NaN otherwise.
    """


    # Ensure needed columns exist
    needed = {'batter_id', 'season', 'level_abbr', 'exit_velo', 'age'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Coerce to numeric and drop NaNs
    work = df.copy()
    work['exit_velo'] = pd.to_numeric(work['exit_velo'], errors='coerce')
    work['age'] = pd.to_numeric(work['age'], errors='coerce')
    work = work.dropna(subset=['exit_velo', 'age'])

    # Aggregate per batter-season-level
    agg = (
        work
        .groupby(['batter_id', 'season', 'level_abbr'], as_index=False)
        .agg(mean_ev=('exit_velo', 'mean'),
             age=('age', 'median'),
             n_bbe=('exit_velo', 'size'))
    )

    # Basic cleaning
    agg = agg.query(f'{age_min} <= age <= {age_max} and n_bbe >= {min_bbe}').copy()
    if agg.empty:
        raise ValueError("No rows left after filtering; relax filters or check data.")
    agg['level_abbr'] = agg['level_abbr'].astype(str).str.lower()

    # Center age (global centering for interpretability)
    age_mean = agg['age'].mean()
    agg['age_c'] = agg['age'] - age_mean

    # Weighted LS with cluster-robust SE by batter
    w = agg['n_bbe'].astype(float)
    model_all = smf.wls('mean_ev ~ age_c + I(age_c**2)', data=agg, weights=w)
    fit_all = model_all.fit(cov_type='cluster', cov_kwds={'groups': agg['batter_id']})

    # Pull coefficients robustly
    b1 = fit_all.params.get('age_c')
    # Patsy usually names this 'I(age_c ** 2)'; fall back to any term containing 'age_c' and '2'
    quad_key = 'I(age_c ** 2)' if 'I(age_c ** 2)' in fit_all.params.index else \
               next((k for k in fit_all.params.index if 'age_c' in k and '2' in k), None)

    if quad_key is None:
        raise KeyError("Quadratic term not found in fitted parameters.")
    b2 = fit_all.params[quad_key]

    # Vertex (peak) only meaningful for concave parabola (b2 < 0)
    if b2 > 0:
        min_age_overall = -b1 / (2 * b2) + age_mean
        print(f"Overall minimum age (quadratic): {min_age_overall:.2f} years")

    print(fit_all.summary())
    if np.isnan(min_age_overall):
        print("Note: quadratic curvature is not concave (b2 >= 0), so no finite 'peak' age.")
    else:
        print(f"Overall peak age (quadratic vertex): {min_age_overall:.2f} years")

    return fit_all, min_age_overall

