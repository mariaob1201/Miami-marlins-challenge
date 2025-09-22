# arima_per_batter_2024_auto.py
import warnings

warnings.filterwarnings("ignore")

from typing import Tuple, Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Import auto_arima from pmdarima
try:
    from pmdarima import auto_arima

    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False
    print("Warning: pmdarima not found. Install with: pip install pmdarima")
    print("Falling back to manual grid search...")

# ----------------------------
# Constants / helpers
# ----------------------------
LEAGUES = ["AA", "AAA", "MLB"]
LEVEL_EQUIV = {"MLB": 0.0, "AAA": -2.0, "AA": -5.0}  # fallback deltas


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ----------------------------
# Angle feature engineering (unchanged)
# ----------------------------
def _wrap180(a):
    """Wrap degrees to [-180, 180]. Works with numpy arrays / pandas Series."""
    return (a + 180.0) % 360.0 - 180.0


def _runlen_stats(mask: pd.Series) -> tuple[int, float]:
    """
    Run-length stats for a boolean mask (e.g., sweet-spot indicator).
    Returns (max_run_length, mean_run_length) for runs of True.
    """
    if mask.empty:
        return 0, 0.0
    m = mask.astype(int).to_numpy()
    # Add 0 at both ends so diff will detect edges
    diff = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    if lengths.size == 0:
        return 0, 0.0
    return int(lengths.max()), float(lengths.mean())


def build_angle_features(df_bbe: pd.DataFrame,
                         season_col="season",
                         la_col="launch_angle",
                         spray_col="spray_angle") -> pd.DataFrame:
    """
    Season-level non-linear features from per-BBE angles.

    Returns a DataFrame indexed by season with (at least) these columns:
      - la_sweet_count: number of BBEs in the launch-angle sweet spot
      - bbe_count: total BBEs
      - la_sweet_rate: share of BBEs with launch angle in [-25, 25] deg
      - la_outside_rate: share of BBEs outside the sweet spot
      - la_dist2: mean squared distance of launch angle from 0 deg
      - la_outside2: mean squared distance outside the sweet spot only (0 inside)
      - la_std: std dev of launch angle
      - la_mad: median absolute deviation of launch angle
      - la_cv: coefficient of variation (std / |mean|) [guarded]
      - la_sweet_run_max: max consecutive "sweet" hits within the season
      - la_sweet_run_mean: mean run length of "sweet" hits within the season
      - spray_cos, spray_sin: circular encoding of spray angle (means)
      - spray_abs_dev: mean absolute deviation |spray|
      - spray_resultant_R: length of mean resultant vector (0..1; higher = more concentrated)
      - spray_circ_var: circular variance = 1 - R

    If angle columns are missing or no usable rows, returns empty DataFrame.
    """
    if la_col not in df_bbe.columns or spray_col not in df_bbe.columns:
        return pd.DataFrame()

    d = df_bbe.dropna(subset=[season_col, la_col, spray_col]).copy()
    if d.empty:
        return pd.DataFrame()

    # --- Launch angle metrics ---
    la = d[la_col].astype(float)
    sweet_min, sweet_max = -25.0, 25.0
    sweet_mask = (la >= sweet_min) & (la <= sweet_max)

    d["_sweet"] = sweet_mask.astype(float)
    d["_la_dist2"] = (la - 0.0) ** 2

    # Outside-only quadratic penalty (0 inside, quadratic outside)
    outside = np.where(la < sweet_min, sweet_min - la,
                       np.where(la > sweet_max, la - sweet_max, 0.0))
    d["_la_out2"] = outside ** 2

    d["_la"] = la  # keep raw for dispersion stats

    # --- Spray angle circular metrics ---
    th_wrapped = _wrap180(d[spray_col].astype(float))
    th_rad = np.deg2rad(th_wrapped)
    d["_c"] = np.cos(th_rad)
    d["_s"] = np.sin(th_rad)
    d["_abs_dev"] = np.abs(th_wrapped)

    # --- Basic season aggregates via .agg ---
    g = d.groupby(season_col, dropna=True).agg(
        la_sweet_count=("_sweet", "sum"),
        bbe_count=("_sweet", "size"),
        la_sweet_rate=("_sweet", "mean"),
        la_dist2=("_la_dist2", "mean"),
        la_outside2=("_la_out2", "mean"),
        la_std=("_la", "std"),
        la_mad=("_la", lambda x: np.median(np.abs(x - np.median(x))) if len(x) else np.nan),
        spray_cos=("_c", "mean"),
        spray_sin=("_s", "mean"),
        spray_abs_dev=("_abs_dev", "mean"),
    )

    # Add outside rate & CV
    g["la_outside_rate"] = 1.0 - g["la_sweet_rate"]
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(np.abs(d.groupby(season_col)["_la"].mean().to_numpy()) > 1e-6,
                         np.abs(d.groupby(season_col)["_la"].mean().to_numpy()),
                         np.nan)
    g["la_cv"] = g["la_std"] / denom

    # Spray concentration from mean cos/sin
    mc = g["spray_cos"].to_numpy()
    ms = g["spray_sin"].to_numpy()
    R = np.sqrt(mc ** 2 + ms ** 2)
    g["spray_resultant_R"] = R
    g["spray_circ_var"] = 1.0 - R

    # --- Sweet-spot streakiness per season (apply custom function) ---
    # Compute per-season run-length stats of sweet hits
    runs = (
        d.groupby(season_col)["_sweet"]
        .apply(lambda s: pd.Series(
            _runlen_stats(s > 0.5),
            index=["la_sweet_run_max", "la_sweet_run_mean"]
        ))
    )
    g = g.join(runs, how="left")

    return g


# ----------------------------
# Build per-batter annual target + exog blocks (unchanged)
# ----------------------------
def build_batter_annual(df: pd.DataFrame,
                        batter_id: str,
                        season_col: str = "season",
                        value_col: str = "exit_velo",
                        level_col: str = "level_abbr",
                        hit_type_col: str = "hit_type"
                        ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      y: Series(float) indexed by annual PeriodIndex, mean exit_velo per season
      shares: DataFrame(float) same index, columns ['AA','AAA','MLB'] (league shares)
      hit_type_shares: DataFrame(float) same index, columns for each hit type (seasonal mean of one-hots)
    Notes:
      - If hit_type_col does not exist, hit_type_shares is empty (not an error).
    """
    cols_needed = [season_col, value_col, level_col]
    if hit_type_col in df.columns:
        cols_needed.append(hit_type_col)

    sub = df.loc[df["batter_id"] == batter_id, cols_needed].copy()
    if sub.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    # Normalize level values
    sub[level_col] = sub[level_col].astype(str).str.strip().str.upper()
    sub = sub[sub[level_col].isin(LEAGUES)]
    if sub.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    # annual mean EV
    y = sub.groupby(season_col)[value_col].mean().astype(float).sort_index()
    if y.index.dtype.kind not in "iu":
        y.index = y.index.astype(int)

    # league counts -> shares per season
    counts = (sub.assign(_cnt=1)
              .groupby([season_col, level_col])["_cnt"].count()
              .unstack(level_col, fill_value=0))
    for col in LEAGUES:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[LEAGUES]

    # hit_type seasonal shares (mean of one-hots)
    if hit_type_col in sub.columns:
        dummies = pd.get_dummies(sub[hit_type_col], prefix=hit_type_col, dtype=float)
        ht = pd.concat([sub[[season_col]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        hit_type_shares = ht.groupby(season_col)[dummies.columns].mean()
    else:
        hit_type_shares = pd.DataFrame()

    # continuous yearly support
    full_years = np.arange(int(y.index.min()), int(y.index.max()) + 1, dtype=int)
    y = y.reindex(full_years)
    counts = counts.reindex(full_years, fill_value=0)

    if not hit_type_shares.empty:
        hit_type_shares = hit_type_shares.reindex(full_years, fill_value=0)
        hit_type_shares = hit_type_shares.interpolate("linear", limit_direction="both")

    # league shares (row sum = 1 where data exist; otherwise 0 then ffill/bfill)
    denom = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.divide(denom, axis=0)
    shares = shares.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # gentle fill for any y gaps from reindex
    if y.isna().any():
        y = y.interpolate("linear", limit_direction="both").fillna(method="ffill").fillna(method="bfill")

    # Use annual PeriodIndex
    y.index = pd.PeriodIndex(y.index.astype(int), freq="Y")
    shares.index = y.index
    if not hit_type_shares.empty:
        hit_type_shares.index = y.index

    return y.astype(float), shares.astype(float), hit_type_shares.astype(float)


def build_exog_data(raw_df: pd.DataFrame,
                    batter_id: str,
                    season_col: str = "season",
                    la_col: str = "launch_angle",
                    spray_col: str = "spray_angle",
                    X_league: pd.DataFrame | None = None,
                    X_hit_type: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge season-level angle, league-share, and hit_type features into a single exog matrix X.
    Index must align with y (PeriodIndex annual). Non-league features are carried forward
    into the 2024 row when forecasting.
    """
    sub = raw_df.loc[raw_df["batter_id"] == batter_id].copy()
    ang = build_angle_features(sub, season_col=season_col, la_col=la_col, spray_col=spray_col)

    pieces = [df for df in [X_league, X_hit_type, ang] if df is not None and not df.empty]
    if not pieces:
        return pd.DataFrame()

    # Start from the first piece, then align-merge others by index
    X = pieces[0].copy()
    for df in pieces[1:]:
        X = pd.concat([X, df.reindex(X.index)], axis=1)

    # Light cleaning
    X = X.interpolate("linear", limit_direction="both")
    X = X.ffill().bfill().fillna(0.0)
    return X


# ----------------------------
# NEW: Auto-ARIMA order selection
# ----------------------------
def choose_auto_arimax(y: pd.Series, X: pd.DataFrame,
                       max_p: int = 3, max_q: int = 3, max_d: int = 2,
                       seasonal: bool = False,
                       information_criterion: str = 'aic',
                       suppress_warnings: bool = True,
                       stepwise: bool = True,
                       **auto_arima_kwargs) -> Dict[str, Any]:
    """
    Use auto_arima to automatically select the best ARIMA order.

    Parameters:
    -----------
    y : pd.Series
        Time series data
    X : pd.DataFrame
        Exogenous variables
    max_p, max_q, max_d : int
        Maximum values for AR, MA, and differencing terms
    seasonal : bool
        Whether to consider seasonal ARIMA models
    information_criterion : str
        'aic', 'bic', or 'hqic'
    stepwise : bool
        Whether to use stepwise algorithm (faster but may miss global optimum)
    **auto_arima_kwargs
        Additional arguments passed to auto_arima

    Returns:
    --------
    Dict with order, aic, and other info
    """

    if not AUTO_ARIMA_AVAILABLE:
        # Fallback to manual grid search with smaller grid
        return choose_arimax_manual(y, X, p_grid=(0, 1, 2), d_grid=(0, 1), q_grid=(0, 1, 2))

    try:
        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter("ignore")

            # Use auto_arima to find the best model
            model = auto_arima(
                y,
                exogenous=X,
                max_p=max_p,
                max_q=max_q,
                max_d=max_d,
                seasonal=seasonal,
                information_criterion=information_criterion,
                suppress_warnings=suppress_warnings,
                stepwise=stepwise,
                error_action='ignore',  # Don't raise errors, return None instead
                **auto_arima_kwargs
            )

            if model is None:
                return {"order": None, "rmse": np.nan, "mae": np.nan, "aic": np.inf,
                        "mode": "auto_arima_fail"}

            order = model.order
            aic = float(model.aic()) if hasattr(model, 'aic') else np.inf

            return {
                "order": order,
                "rmse": np.nan,  # auto_arima doesn't provide CV scores
                "mae": np.nan,
                "aic": aic,
                "mode": "auto_arima",
                "model": model  # Store the fitted model
            }

    except Exception as e:
        print(f"Auto-ARIMA failed: {e}")
        # Fallback to manual method
        return choose_arimax_manual(y, X, p_grid=(0, 1), d_grid=(0, 1), q_grid=(0, 1))


# ----------------------------
# Manual grid search (fallback)
# ----------------------------
def rolling_origin(y: pd.Series, X: pd.DataFrame, initial: int, horizon: int = 1):
    n = len(y)
    if n != len(X):
        raise ValueError("y and X must have same length.")
    for col in LEAGUES:
        if col not in X.columns:
            raise ValueError(f"Missing league exog column '{col}' in X.")
    if initial + horizon >= n:
        raise ValueError(
            f"Not enough annual points for rolling validation: initial={initial}, horizon={horizon}, n={n}")
    for t in range(initial, n - horizon):
        yield y.iloc[:t], X.iloc[:t, :], y.iloc[t:t + horizon], X.iloc[t:t + horizon, :]


def validate_order(y: pd.Series, X: pd.DataFrame, order: Tuple[int, int, int],
                   initial: int, horizon: int = 1) -> Dict[str, float]:
    preds, trues, aics = [], [], []
    for ty, tX, vy, vX in rolling_origin(y, X, initial=initial, horizon=horizon):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                res = ARIMA(ty, order=order, exog=tX,
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit()
            fc = res.get_forecast(steps=horizon, exog=vX).predicted_mean.values
        except Exception:
            return {"rmse": np.inf, "mae": np.inf, "aic": np.inf}
        preds.append(fc[-1]);
        trues.append(vy.iloc[-1])
        aics.append(res.aic if np.isfinite(res.aic) else np.inf)
    return {"rmse": rmse(trues, preds),
            "mae": float(np.mean(np.abs(np.array(trues) - np.array(preds)))),
            "aic": float(np.nanmean(aics))}


def choose_arimax_manual(y: pd.Series, X: pd.DataFrame,
                         p_grid=(0, 1), d_grid=(0, 1), q_grid=(0, 1),
                         min_initial_years: int = 5,
                         cv_horizon: int = 1) -> Dict[str, Any]:
    """
    Original manual grid search method (unchanged from original code)
    """
    n = len(y)
    max_allowed_initial = n - cv_horizon - 1
    adaptive_initial = max(3, min(min_initial_years, max_allowed_initial)) if max_allowed_initial >= 3 else None

    # CV path
    if adaptive_initial is not None:
        results = []
        for p in p_grid:
            for d in d_grid:
                for q in q_grid:
                    order = (p, d, q)
                    scores = validate_order(y, X, order, initial=adaptive_initial, horizon=cv_horizon)
                    results.append({"order": order, **scores})
        df = pd.DataFrame(results).sort_values(["rmse", "aic"])
        best = df.iloc[0].to_dict()
        return best | {"mode": "cv_manual", "leaderboard": df, "initial_used": adaptive_initial, "horizon": cv_horizon}

    # AIC-only path
    aic_rows = []
    for p in p_grid:
        for d in d_grid:
            for q in q_grid:
                order = (p, d, q)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        res = ARIMA(y, order=order, exog=X,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()
                    aic_rows.append({"order": order, "aic": float(res.aic)})
                except Exception:
                    continue
    if not aic_rows:
        return {"order": None, "rmse": np.nan, "mae": np.nan, "aic": np.inf,
                "mode": "fail", "initial_used": None, "horizon": cv_horizon}
    best = min(aic_rows, key=lambda r: r["aic"])
    return {"order": best["order"], "rmse": np.nan, "mae": np.nan, "aic": float(best["aic"]),
            "mode": "aic_manual", "initial_used": None, "horizon": cv_horizon}


# ----------------------------
# Forecast helpers (unchanged)
# ----------------------------
def make_exog_2024_row(X_hist: pd.DataFrame,
                       target_level: str,
                       leagues: List[str] = LEAGUES) -> pd.DataFrame:
    """
    Build a 1-row exog for 2024 by:
      - copying the last row of X (carry-forward all non-league features)
      - zeroing league columns and setting the requested target level to 1
      - advancing index by 1 annual period
    """
    cols = X_hist.columns.tolist()
    last_row = X_hist.iloc[[-1]].copy()
    for L in leagues:
        if L in last_row.columns:
            last_row[L] = 1.0 if L == target_level else 0.0
    next_idx = X_hist.index[-1] + 1
    last_row.index = [next_idx]
    return last_row[cols]


# ----------------------------
# Final fit on full history + 2024 forecast
# ----------------------------
def refit_and_forecast_2024(y: pd.Series, X: pd.DataFrame, order: Tuple[int, int, int],
                            target_level: str, fitted_model=None) -> Dict[str, float]:
    """
    Modified to optionally use pre-fitted auto_arima model
    """
    if fitted_model is not None and AUTO_ARIMA_AVAILABLE:
        # Use the already fitted auto_arima model
        exog_2024 = make_exog_2024_row(X, target_level=target_level, leagues=LEAGUES)
        try:
            fc = fitted_model.predict(n_periods=1, exogenous=exog_2024.values)
            conf_int = fitted_model.predict(n_periods=1, exogenous=exog_2024.values,
                                            return_conf_int=True, alpha=0.05)[1]
            mean = float(fc[0])
            lower, upper = float(conf_int[0, 0]), float(conf_int[0, 1])
            return {"forecast": mean, "lower_95": lower, "upper_95": upper}
        except Exception:
            # Fall back to manual refit
            pass

    # Manual refit with statsmodels ARIMA
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
    return {"forecast": mean, "lower_95": lower, "upper_95": upper}


def fallback_last_plus_level(y: pd.Series, last_level: str, target_level: str) -> Dict[str, float]:
    last_val = float(y.iloc[-1])
    delta = LEVEL_EQUIV.get(target_level, 0.0) - LEVEL_EQUIV.get(last_level, 0.0)
    pred = last_val + delta
    return {"forecast": pred, "lower_95": np.nan, "upper_95": np.nan}


# ----------------------------
# MODIFIED: End-to-end per-batter 2024 with auto-ARIMA
# ----------------------------
def predict_2024_per_batter(csv_path: str,
                            season_col="season",
                            value_col="exit_velo",
                            level_col="level_abbr",
                            target_level_mode: str = "last",  # 'last' or 'all'
                            use_auto_arima: bool = True,  # NEW parameter
                            max_p: int = 3, max_q: int = 3, max_d: int = 2,  # auto-ARIMA params
                            p_grid=(0, 1), d_grid=(0, 1), q_grid=(0, 1),  # manual fallback params
                            min_initial_years=5,
                            output_csv="per_batter_2024_auto_arimax.csv") -> pd.DataFrame:
    la_col = "launch_angle"
    spray_col = "spray_angle"

    raw = pd.read_csv(csv_path)
    raw = raw.dropna(subset=["batter_id", season_col, value_col, level_col]).copy()
    raw[season_col] = pd.to_numeric(raw[season_col], errors="coerce")
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna(subset=[season_col, value_col])

    # normalize level strings early for consistency
    raw[level_col] = raw[level_col].astype(str).str.strip().str.upper()

    batters = raw["batter_id"].dropna().unique().tolist()
    out_rows: List[Dict[str, Any]] = []

    print(
        f"Using {'Auto-ARIMA' if use_auto_arima and AUTO_ARIMA_AVAILABLE else 'Manual Grid Search'} for model selection")

    for i, b in enumerate(batters, 1):
        if i % 100 == 0:
            print(f"  ... {i}/{len(batters)}")

        y, shares, hit_type_shares = build_batter_annual(
            raw, b, season_col, value_col, level_col, hit_type_col="hit_type"
        )
        # need at least a short history
        if y.empty or shares.empty or len(y) < 3:
            continue

        # build exogenous matrix (league + optional hit_type + optional angles)
        X = build_exog_data(
            raw_df=raw,
            batter_id=b,
            season_col=season_col,
            la_col=la_col,
            spray_col=spray_col,
            X_league=shares,
            X_hit_type=hit_type_shares if not hit_type_shares.empty else None
        )
        if X.empty:
            # (should be rare; proceed with league-only)
            X = shares.copy()

        # last observed season & level
        last_year = int(y.index[-1].year)
        last_rows = raw[(raw["batter_id"] == b) & (raw[season_col] == last_year)]
        last_level = (last_rows[level_col].mode().iloc[0] if not last_rows.empty else "MLB")
        if last_level not in LEAGUES:
            last_level = "MLB"

        targets = [last_level] if target_level_mode == "last" else LEAGUES

        # choose order using auto-ARIMA or manual method
        if use_auto_arima and AUTO_ARIMA_AVAILABLE:
            best = choose_auto_arimax(
                y, X,
                max_p=max_p,
                max_q=max_q,
                max_d=max_d,
                seasonal=False,  # Usually False for annual data
                stepwise=True
            )
        else:
            best = choose_arimax_manual(
                y, X,
                p_grid=p_grid,
                d_grid=d_grid,
                q_grid=q_grid,
                min_initial_years=min_initial_years,
                cv_horizon=1
            )

        can_fit = (best["order"] is not None) and (
                np.isfinite(best.get("rmse", np.nan)) or
                best.get("mode") in ["aic_manual", "auto_arima"]
        )

        for L in targets:
            if can_fit:
                try:
                    # Pass the fitted model if available from auto_arima
                    fitted_model = best.get("model") if use_auto_arima else None
                    pred = refit_and_forecast_2024(y, X, tuple(int(x) for x in best["order"]), L, fitted_model)

                    method_name = f"Auto-ARIMAX{tuple(int(x) for x in best['order'])}" if use_auto_arima else f"ARIMAX{tuple(int(x) for x in best['order'])}"

                    out_rows.append({
                        "batter_id": b,
                        "last_season": last_year,
                        "last_level": last_level,
                        "target_level_2024": L,
                        "method": method_name,
                        "fit_mode": best.get("mode"),
                        "val_rmse": float(best.get("rmse", np.nan)),
                        "val_mae": float(best.get("mae", np.nan)),
                        "val_aic": float(best.get("aic", np.nan)),
                        **pred
                    })
                    continue
                except Exception as e:
                    print(f"Forecast failed for batter {b}, level {L}: {e}")
                    pass  # fall through to fallback

            # fallback
            fb = fallback_last_plus_level(y, last_level=last_level, target_level=L)
            out_rows.append({
                "batter_id": b,
                "last_season": last_year,
                "last_level": last_level,
                "target_level_2024": L,
                "method": "fallback_last+level_delta",
                "fit_mode": best.get("mode"),
                "val_rmse": float(best.get("rmse", np.nan)),
                "val_mae": float(best.get("mae", np.nan)),
                "val_aic": float(best.get("aic", np.nan)),
                **fb
            })

    out = pd.DataFrame(out_rows).sort_values(["batter_id", "target_level_2024"]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    path = '/Users/mariaoros/Documents/Github-projects/2025/Miami-marlins-challenge/3_Modeling/'
    out.to_csv(path+'time_series/outputs/per_batter_2024_auto_arimax2.csv', index=False)
    print(f"\nSaved 2024 per-batter forecasts to: {output_csv}")
    return out


if __name__ == "__main__":
    _ = predict_2024_per_batter(
        csv_path="1_Data/exit_velo_project_data.csv",
        season_col="season",
        value_col="exit_velo",
        level_col="level_abbr",
        target_level_mode="all",  # or "last" for last level only
        use_auto_arima=True,  # NEW: Use auto-ARIMA instead of grid search
        max_p=3, max_q=3, max_d=2,  # Auto-ARIMA search bounds
        p_grid=(0, 1), d_grid=(0, 1), q_grid=(0, 1),  # Fallback manual grid
        min_initial_years=5,
        output_csv="3_Modeling/time_series/outputs/per_batter_2024_auto_arimax.csv"
    )