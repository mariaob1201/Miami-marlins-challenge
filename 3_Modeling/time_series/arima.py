'''
INSTRUCTIONS TO RUN THIS CODE

Please be sure that the input data exist eg exit_velo_project_data.csv in a folder 1_Data
The structure of my code is
1_Data
    \csv data with the original observations and validation datasets
2_Data_Exploration
    conteins the data_exploration.ipynb that is attached as derivable
3_Modeling
    time_series/
        outputs/
        arima.py #this code
        rev.ipynb # one of the derivables
        forecasting2024_mlb_maria.csv #this is the output from rev.ipynb


My Github repo contains other code that was used to explore potential solutions, including PyMC under a hierarchical framework accoutning for within and between player variance and accounting for fixed and random effects. Neverthless, for time constraints, I am not including such effort in this code but can be explored in the github repo.


Maria Oros
'''


import warnings

warnings.filterwarnings("ignore")

from typing import Tuple, Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pmdarima as pm

LEAGUES = ["AA", "AAA", "MLB"]
LEVEL_EQUIV = {"MLB": 0.0, "AAA": -2.5, "AA": -4.5}

root = ''#path where source data is located

def rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Square Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =========================
# Angle feature engineering
# =========================
def _wrap180(a: float) -> float:
    """Wrap degrees to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def build_angle_features(df_bbe: pd.DataFrame,
                         season_col: str = "season",
                         la_col: str = "launch_angle",
                         spray_col: str = "spray_angle") -> pd.DataFrame:
    """
    Season-level non-linear features derived from per-BBE angles.
    Returns DataFrame with angle-based features by season.
    """
    d = df_bbe.dropna(subset=[season_col]).copy()
    if la_col not in d.columns or spray_col not in d.columns:
        return pd.DataFrame()

    d = d.dropna(subset=[la_col, spray_col])
    if d.empty:
        return pd.DataFrame()

    # Launch angle features
    la = d[la_col].astype(float)
    sa = d[spray_col].astype(float)

    ## launch and spray as per exploration
    sweet = ((la >= 2.25) & (la <= 21.9) & (sa >= -2.7) & (sa <= 12.9)).astype(float)
    la_dist2 = (la - 12.0) ** 2  # Added missing variable

    # Spray angle features
    th_deg = d[spray_col].astype(float)
    th_wrapped = th_deg.apply(_wrap180)
    th_rad = np.deg2rad(th_wrapped)
    c = np.cos(th_rad)
    s = np.sin(th_rad)
    abs_dev = np.abs(th_wrapped)

    d = d.assign(
        _sweet=sweet,
        _la_dist2=la_dist2,
        _c=c,
        _s=s,
        _abs_dev=abs_dev
    )

    # Aggregate by season in temrs of how muh the player hits the sweet
    g = d.groupby(season_col).agg(
        la_sweet_rate=("_sweet", "mean"),
        la_dist2=("_la_dist2", "mean"),  # Added missing aggregation
        spray_cos=("_c", "mean"),
        spray_sin=("_s", "mean"),
        spray_abs_dev=("_abs_dev", "mean")
    ).sort_index()

    if g.index.dtype.kind not in "iu":
        g.index = g.index.astype(int)
        g = g.sort_index()

    return g.astype(float)


# ===========================================
# Build per-batter annual data
# ===========================================
def build_batter_annual(df: pd.DataFrame,
                        batter_id: str,
                        season_col: str = "season",
                        value_col: str = "exit_velo",
                        level_col: str = "level_abbr",
                        hit_type_col: str = "hit_type") -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Build annual time series data for a batter.
    Returns: (target_series, league_shares, hit_type_shares)
    """
    # Get player data
    cols_needed = [season_col, value_col, level_col, hit_type_col]
    sub = df.loc[df["batter_id"] == batter_id, cols_needed].copy()
    if sub.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    # Normalize level names
    sub[level_col] = sub[level_col].astype(str).str.strip().str.upper()

    sub = sub[sub[level_col].isin(LEAGUES)]
    if sub.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    # Build annual exit velocity target
    y = sub.groupby(season_col)[value_col].mean().astype(float).sort_index()
    hit_type_dummies = pd.get_dummies(sub[hit_type_col], prefix=hit_type_col, dtype=float)
    sub_with_dummies = pd.concat([sub, hit_type_dummies], axis=1)
    hit_type_shares = sub_with_dummies.groupby(season_col)[hit_type_dummies.columns].mean()

    # Build league shares
    counts = (sub.assign(_cnt=1)
              .groupby([season_col, level_col])["_cnt"]
              .count()
              .unstack(level_col, fill_value=0))

    # Ensure all leagues present
    for col in LEAGUES:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[LEAGUES]

    # Create continuous time series
    full_years = np.arange(int(y.index.min()), int(y.index.max()) + 1, dtype=int)
    y = y.reindex(full_years)
    counts = counts.reindex(full_years, fill_value=0)
    hit_type_shares = hit_type_shares.reindex(full_years, fill_value=0)

    # Calculate league shares
    denom = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.divide(denom, axis=0)
    shares = shares.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # Fill missing values, probably to perform later, I dont like to much the filling
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
        y = y.fillna(method="ffill").fillna(method="bfill")

    hit_type_shares = hit_type_shares.interpolate(method="linear", limit_direction="both")

    # Convert to period index
    y.index = pd.PeriodIndex(y.index.astype(int), freq="Y")
    shares.index = y.index
    hit_type_shares.index = y.index

    return y.astype(float), shares.astype(float), hit_type_shares.astype(float)


# ==========================
# Model validation and selection
# ==========================
def rolling_origin(y: pd.Series, X: pd.DataFrame, initial: int, horizon: int = 1):
    """Generate rolling origin cross-validation splits"""
    n = len(y)
    if n != len(X):
        raise ValueError("y and X must have same length.")
    if initial + horizon >= n:
        raise ValueError(f"Not enough annual points: initial={initial}, horizon={horizon}, n={n}")

    for t in range(initial, n - horizon):
        yield y.iloc[:t], X.iloc[:t, :], y.iloc[t:t + horizon], X.iloc[t:t + horizon, :]


def validate_auto_arimax(y: pd.Series, X: pd.DataFrame, initial: int, horizon: int = 1) -> Dict[str, Any]:
    """Cross-validate auto-ARIMAX model"""
    preds, trues, aics = [], [], []
    best_order = (0, 0, 0)

    for ty, tX, vy, vX in rolling_origin(y, X, initial=initial, horizon=horizon):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                fit = pm.auto_arima(
                    ty, X=tX,
                    start_p=0, start_q=0,
                    max_p=3, max_q=3,
                    m=1, d=None, D=0,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
            res = fit.fit(ty, X=tX)
            fc, conf_int = res.predict(n_periods=horizon, X=vX, return_conf_int=True)

            preds.append(fc.values[-1])
            trues.append(vy.iloc[-1])
            aics.append(res.aic() if np.isfinite(res.aic()) else np.inf)
            best_order = fit.order

        except Exception:
            return {"rmse": np.inf, "mae": np.inf, "aic": np.inf, "order": None}

    return {
        "rmse": rmse(trues, preds),
        "mae": float(np.mean(np.abs(np.array(trues) - np.array(preds)))),
        "aic": float(np.nanmean(aics)),
        "order": best_order
    }


def choose_arimax(y: pd.Series, X: pd.DataFrame,
                  min_initial_years: int = 2,
                  cv_horizon: int = 1) -> Dict[str, Any]:
    """Choose best ARIMAX model using cross-validation or AIC"""

    # If X is empty or has no variation, use simple ARIMA
    if X.empty or X.shape[1] == 0:
        print(f"    No valid exogenous variables, using simple ARIMA")
        return choose_simple_arima(y, cv_horizon)

    n = len(y)
    max_allowed_initial = n - cv_horizon - 1
    adaptive_initial = max(2, min(min_initial_years, max_allowed_initial)) if max_allowed_initial >= 2 else None

    # Try cross-validation first
    if adaptive_initial is not None:
        try:
            scores = validate_auto_arimax(y, X, initial=adaptive_initial, horizon=cv_horizon)
            if scores["order"] is not None:
                return scores | {"mode": "cv", "initial_used": adaptive_initial, "horizon": cv_horizon}
        except Exception as e:
            print(f"    Cross-validation failed, trying AIC-only: {e}")

    # Fall back to AIC-only selection
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fit = pm.auto_arima(
                y, X=X,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                m=1, d=None, D=0,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
        best_order = fit.order
        res = fit.fit(y, X=X)
        aic = float(res.aic()) if np.isfinite(res.aic()) else np.inf

        return {
            "order": best_order,
            "rmse": np.nan,
            "mae": np.nan,
            "aic": aic,
            "mode": "aic",
            "initial_used": None,
            "horizon": cv_horizon
        }
    except Exception as e:
        print(f"    ARIMAX failed, falling back to simple ARIMA: {e}")
        return choose_simple_arima(y, cv_horizon)


def choose_simple_arima(y: pd.Series, cv_horizon: int = 1) -> Dict[str, Any]:
    """Fallback to simple ARIMA without exogenous variables"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fit = pm.auto_arima(
                y,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                m=1, d=None, D=0,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
        best_order = fit.order
        res = fit.fit(y)
        aic = float(res.aic()) if np.isfinite(res.aic()) else np.inf

        return {
            "order": best_order,
            "rmse": np.nan,
            "mae": np.nan,
            "aic": aic,
            "mode": "simple_arima",
            "initial_used": None,
            "horizon": cv_horizon
        }
    except Exception:
        return {
            "order": None,
            "rmse": np.nan,
            "mae": np.nan,
            "aic": np.inf,
            "mode": "fail",
            "initial_used": None,
            "horizon": cv_horizon
        }


# ===============================================
# Forecasting
# ===============================================
def build_exog_data(raw_df: pd.DataFrame,
                    batter_id: str,
                    season_col: str = "season",
                    la_col: str = "launch_angle",
                    spray_col: str = "spray_angle",
                    X_league: pd.DataFrame = None,
                    X_hit_type: pd.DataFrame = None) -> pd.DataFrame:
    """Build combined exogenous feature matrix"""
    # Get angle features
    sub = raw_df.loc[raw_df["batter_id"] == batter_id].copy()
    ang = build_angle_features(sub, season_col=season_col, la_col=la_col, spray_col=spray_col)

    # Combine all available feature sets
    exog_list = [df for df in [X_league, X_hit_type, ang] if df is not None and not df.empty]

    if not exog_list:
        return pd.DataFrame()

    # Start with first feature set and align others
    X = exog_list[0].copy()
    for df in exog_list[1:]:
        aligned_df = df.reindex(X.index)
        X = pd.concat([X, aligned_df], axis=1)

    # Fill missing values
    X = X.interpolate(method="linear", limit_direction="both")
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # Drop columns that are not predictive eg constant (no variation)
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:  # Column has 0 or 1 unique values
            constant_cols.append(col)

    if constant_cols:
        print(f"  Removing constant columns for player {batter_id}: {constant_cols}")
        X = X.drop(columns=constant_cols)

    # If we have league columns, drop one to avoid perfect multicollinearity
    # (n-1 dummy encoding instead of n dummy encoding)
    league_cols = [col for col in X.columns if col in LEAGUES]
    if len(league_cols) > 1:
        # Drop the last league column (reference category)
        reference_league = league_cols[-1]
        X = X.drop(columns=[reference_league])
        if len(X.columns) > 0:  # Only print if we still have columns
            print(f"  Dropped reference league '{reference_league}' for player {batter_id}")

    return X


def make_exog_2024_row(X_hist: pd.DataFrame,
                       target_level: str,
                       leagues: List[str] = LEAGUES) -> pd.DataFrame:
    """Create exogenous features for 2024 forecast"""
    last_row = X_hist.iloc[[-1]].copy()

    # Project angle features using recent trend
    for col in [c for c in X_hist.columns if c not in leagues]:
        if len(X_hist) >= 3:
            recent_trend = X_hist[col].iloc[-3:].diff().mean()
            if np.isfinite(recent_trend):
                last_row[col] = last_row[col].iloc[0] + recent_trend

    # Set league indicators (only for leagues present in X_hist), remeber that one category is reference
    available_leagues = [L for L in leagues if L in X_hist.columns]

    for L in available_leagues:
        if L == target_level:
            last_row[L] = 1.0
        else:
            last_row[L] = 0.0

    # Set index for next year
    next_idx = X_hist.index[-1] + 1
    last_row.index = [next_idx]

    return last_row[X_hist.columns]


def refit_and_forecast_2024(y: pd.Series, X: pd.DataFrame, order: Tuple[int, int, int],
                            target_level: str) -> Dict[str, float]:
    """Refit model on full data and forecast 2024"""

    # If X is empty or has no variation, use simple ARIMA without exog
    if X.empty or X.shape[1] == 0:
        print(f"    No exogenous variables available, using simple ARIMA")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            res = ARIMA(y, order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit()

        fc = res.get_forecast(steps=1)
        mean = float(fc.predicted_mean.values[0])
        ci = fc.conf_int(alpha=0.05)
        lower, upper = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])

        return {
            "forecast": mean,
            "lower_95": lower,
            "upper_95": upper,
            "prediction_std": float((upper - lower) / (2 * 1.96))
        }

    # Use ARIMAX with exogenous variables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        res = ARIMA(y, order=order, exog=X,
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit()

    exog_2024 = make_exog_2024_row(X, target_level=target_level, leagues=LEAGUES)
    fc = res.get_forecast(steps=1, exog=exog_2024)

    mean = float(fc.predicted_mean.values[0])
    ci = fc.conf_int(alpha=0.05)
    lower, upper = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])

    return {
        "forecast": mean,
        "lower_95": lower,
        "upper_95": upper,
        "prediction_std": float((upper - lower) / (2 * 1.96))
    }


def fallback_last_plus_level(y: pd.Series, last_level: str, target_level: str) -> Dict[str, float]:
    """Fallback prediction using last value plus level adjustment"""
    last_val = float(y.iloc[-1])
    level_delta = LEVEL_EQUIV.get(target_level, 0.0) - LEVEL_EQUIV.get(last_level, 0.0)
    pred = last_val + level_delta

    # Estimate uncertainty from historical
    if len(y) > 1:
        diffs = y.diff().dropna()
        if len(diffs) > 1:
            std_dev = float(diffs.std())
            uncertainty = 1.96 * std_dev
            return {
                "forecast": pred,
                "lower_95": pred - uncertainty,
                "upper_95": pred + uncertainty,
                "prediction_std": std_dev
            }

    return {
        "forecast": pred,
        "lower_95": np.nan,
        "upper_95": np.nan,
        "prediction_std": np.nan
    }


# ===========================================
# Main prediction function
# ===========================================
def predict_2024_per_batter(csv_path: str,
                            season_col: str = "season",
                            value_col: str = "exit_velo",
                            level_col: str = "level_abbr",
                            hit_type_col: str = "hit_type",
                            la_col: str = "launch_angle",
                            spray_col: str = "spray_angle",
                            target_level_mode: str = "all",
                            min_initial_years: int = 2,
                            min_seasons: int = 3,
                            output_csv: str = "per_batter_2024_predictions.csv") -> pd.DataFrame:
    """
    Generate 2024 predictions for all eligible batters
    """

    # Load and clean data
    print(f"Loading data from {csv_path}...")
    raw1 = pd.read_csv(csv_path)
    eval = pd.read_csv(root+'1_Data/exit_velo_validate_data.csv')
    raw = raw1[raw1['batter_id'].isin(eval['batter_id'].unique())]

    # Basic data cleaning
    required_cols = ["batter_id", season_col, value_col, level_col]
    raw = raw.dropna(subset=required_cols).copy()
    raw[season_col] = pd.to_numeric(raw[season_col], errors="coerce")
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna(subset=[season_col, value_col])

    # Normalize level names
    raw[level_col] = raw[level_col].astype(str).str.strip().str.upper()
    raw[level_col] = raw[level_col].replace({
        "TRIPLE-A": "AAA", "DOUBLE-A": "AA", "MAJOR LEAGUE": "MLB", "ML": "MLB"
    })

    # Filter eligible batters
    batter_summary = raw.groupby("batter_id")[season_col].nunique()
    min_seasons = 1
    eligible_batters = batter_summary[batter_summary >= min_seasons].index.tolist()

    print(f"Processing {len(eligible_batters)} batters with >= {min_seasons} seasons...")

    out_rows: List[Dict[str, Any]] = []
    processing_stats = {"processed": 0, "successful": 0, "fallback": 0, "failed": 0}

    for i, batter_id in enumerate(eligible_batters, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(eligible_batters)} ({i / len(eligible_batters) * 100:.1f}%)")

        processing_stats["processed"] += 1

        try:
            # Build annual data
            y, shares, hit_type_shares = build_batter_annual(
                raw, batter_id, season_col, value_col, level_col, hit_type_col
            )

            if y.empty or shares.empty or len(y) < min_seasons:
                processing_stats["failed"] += 1
                continue

            # Build feature matrix
            X = build_exog_data(
                raw_df=raw,
                batter_id=batter_id,
                season_col=season_col,
                la_col=la_col,
                spray_col=spray_col,
                X_league=shares,
                X_hit_type=hit_type_shares
            )

            if X.empty:
                X = shares.copy()  # Fallback to league shares only

            # Determine last level and targets
            last_year = int(y.index[-1].year)
            last_rows = raw[(raw["batter_id"] == batter_id) & (raw[season_col] == last_year)]
            last_level = (last_rows[level_col].mode().iloc[0] if not last_rows.empty else "MLB")
            if last_level not in LEAGUES:
                last_level = "MLB"

            targets = [last_level] if target_level_mode == "last" else LEAGUES

            # Model selection
            best = choose_arimax(y, X, min_initial_years=min_initial_years, cv_horizon=1)
            can_fit = (best["order"] is not None) and (
                    np.isfinite(best.get("rmse", np.nan)) or best.get("mode") == "aic"
            )

            # Generate predictions for each target level
            for target_level in targets:
                base_row = {
                    "batter_id": batter_id,
                    "last_season": last_year,
                    "last_level": last_level,
                    "target_level_2024": target_level,
                    "seasons_in_data": len(y),
                    "total_observations": len(raw[raw["batter_id"] == batter_id]),
                    "recent_performance": float(y.iloc[-1])
                }

                if can_fit:
                    try:
                        pred = refit_and_forecast_2024(
                            y, X, tuple(int(x) for x in best["order"]), target_level
                        )

                        model_order = tuple(int(x) for x in best['order'])
                        out_rows.append({
                            **base_row,
                            "method": f"ARIMAX{model_order}",
                            "fit_mode": best.get("mode"),
                            "val_rmse": float(best.get("rmse", np.nan)),
                            "val_mae": float(best.get("mae", np.nan)),
                            "val_aic": float(best.get("aic", np.nan)),
                            "arima_order_p": model_order[0],
                            "arima_order_d": model_order[1],
                            "arima_order_q": model_order[2],
                            **pred
                        })
                        processing_stats["successful"] += 1
                        continue

                    except Exception as e:
                        print(f"  Model forecast failed for {batter_id} at {target_level}: {e}")

                # Fallback prediction
                fb = fallback_last_plus_level(y, last_level=last_level, target_level=target_level)
                out_rows.append({
                    **base_row,
                    "method": "fallback_last_plus_level",
                    "fit_mode": best.get("mode", "fallback"),
                    "val_rmse": float(best.get("rmse", np.nan)),
                    "val_mae": float(best.get("mae", np.nan)),
                    "val_aic": float(best.get("aic", np.nan)),
                    "arima_order_p": np.nan,
                    "arima_order_d": np.nan,
                    "arima_order_q": np.nan,
                    **fb
                })
                processing_stats["fallback"] += 1

        except Exception as e:
            print(f"  Error processing batter {batter_id}: {e}")
            processing_stats["failed"] += 1
            continue

    # Create output DataFrame
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["batter_id", "target_level_2024"]).reset_index(drop=True)

    # Print summary
    print(f"\nProcessing Summary:")
    print(f"  Total batters processed: {processing_stats['processed']}")
    print(f"  Successful ARIMAX predictions: {processing_stats['successful']}")
    print(f"  Fallback predictions: {processing_stats['fallback']}")
    print(f"  Failed: {processing_stats['failed']}")
    print(f"  Total predictions generated: {len(out)}")

    # Save results
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")

    return out


# =========================
# Simplified reporting
# =========================
def generate_summary_report(predictions_df: pd.DataFrame, output_dir: str = root+"3_Modeling/time_series/outputs/"):
    """Generate a simple summary report"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Basic statistics
    stats = {
        "total_predictions": len(predictions_df),
        "unique_batters": predictions_df["batter_id"].nunique(),
        "arimax_predictions": len(predictions_df[predictions_df["method"].str.contains("ARIMAX", na=False)]),
        "fallback_predictions": len(predictions_df[predictions_df["method"] == "fallback_last_plus_level"]),
        "avg_forecast": predictions_df["forecast"].mean(),
        "forecast_std": predictions_df["forecast"].std(),
        "forecast_range": [predictions_df["forecast"].min(), predictions_df["forecast"].max()]
    }

    # Method breakdown
    method_counts = predictions_df["method"].value_counts()

    # Target level breakdown
    level_counts = predictions_df["target_level_2024"].value_counts()

    # Save summary
    summary_text = f"""
        PREDICTION SUMMARY REPORT
        ========================
        
        Basic Statistics:
        - Total predictions: {stats['total_predictions']:,}
        - Unique batters: {stats['unique_batters']:,}
        - ARIMAX predictions: {stats['arimax_predictions']:,}
        - Fallback predictions: {stats['fallback_predictions']:,}
        
        Forecast Statistics:
        - Average forecast: {stats['avg_forecast']:.1f} mph
        - Standard deviation: {stats['forecast_std']:.1f} mph
        - Range: [{stats['forecast_range'][0]:.1f}, {stats['forecast_range'][1]:.1f}] mph
        
        Method Breakdown:
        {method_counts.to_string()}
        
        Target Level Breakdown:
        {level_counts.to_string()}
        """

    with open(Path(output_dir) / "summary_report.txt", "w") as f:
        f.write(summary_text)

    # Save top performers
    top_forecasts = predictions_df.nlargest(20, "forecast")[
        ["batter_id", "target_level_2024", "forecast", "lower_95", "upper_95", "method"]
    ]
    top_forecasts.to_csv(Path(output_dir) / "top_forecasts.csv", index=False)

    print(f"Summary report saved to: {output_dir}")
    return stats


if __name__ == "__main__":
    # Main execution
    predictions = predict_2024_per_batter(
        csv_path="1_Data/exit_velo_project_data.csv",
        season_col="season",
        value_col="exit_velo",
        level_col="level_abbr",
        hit_type_col="hit_type",
        la_col="launch_angle",
        spray_col="spray_angle",
        target_level_mode="all",  # or "last" for faster processing
        min_initial_years=2,
        min_seasons=3,
        output_csv=root+"3_Modeling/time_series/outputs/per_batter_2024_predictions.csv"
    )

    # Generate summary report
    if not predictions.empty:
        generate_summary_report(predictions, root+"3_Modeling/time_series/outputs/")
        print("\nPrediction process completed successfully!")
    else:
        print("\nNo predictions were generated. Check data and parameters.")