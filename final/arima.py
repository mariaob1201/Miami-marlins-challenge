# arima_with_angles_and_report_fixed.py

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

# ----------------------------
# Constants / helpers
# ----------------------------
LEAGUES = ["AA", "AAA", "MLB"]
LEVEL_EQUIV = {"MLB": 0.0, "AAA": -2.5, "AA": -4.5}  # fallback deltas


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =========================
# Angle feature engineering
# =========================
def _wrap180(a: float) -> float:
    """Wrap degrees to [-180, 180]."""
    w = (a + 180.0) % 360.0 - 180.0
    return w


def build_angle_features(df_bbe: pd.DataFrame,
                         season_col: str = "season",
                         la_col: str = "launch_angle",
                         spray_col: str = "spray_angle") -> pd.DataFrame:
    """
    Season-level non-linear features derived from per-BBE angles.
    Returns a DataFrame indexed by season with columns:
      la_sweet_rate, la_dist2, spray_cos, spray_sin, spray_abs_dev
    """
    d = df_bbe.dropna(subset=[season_col]).copy()
    if la_col not in d.columns or spray_col not in d.columns:
        return pd.DataFrame()

    d = d.dropna(subset=[la_col, spray_col])
    if d.empty:
        return pd.DataFrame()

    la = d[la_col].astype(float)
    sweet = ((la >= 8.0) & (la <= 32.0)).astype(float)
    la_dist2 = (la - 20.0) ** 2

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

    g = d.groupby(season_col).agg(
        la_sweet_rate=("_sweet", "mean"),
        la_dist2=("_la_dist2", "mean"),
        spray_cos=("_c", "mean"),
        spray_sin=("_s", "mean"),
        spray_abs_dev=("_abs_dev", "mean")
    ).sort_index()

    if g.index.dtype.kind not in "iu":
        g.index = g.index.astype(int)
        g = g.sort_index()

    return g.astype(float)


def add_angle_exog_to_X(raw_df: pd.DataFrame,
                        batter_id: str,
                        season_col: str = "season",
                        la_col: str = "launch_angle",
                        spray_col: str = "spray_angle",
                        X_league: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge season-level angle features with league-share exog.
    Returns X with columns: ['AA','AAA','MLB','la_sweet_rate','la_dist2','spray_cos','spray_sin','spray_abs_dev']
    """
    sub = raw_df.loc[raw_df["batter_id"] == batter_id].copy()
    ang = build_angle_features(sub, season_col=season_col,
                               la_col=la_col, spray_col=spray_col)

    if X_league is None or X_league.empty:
        X = ang
    else:
        ang = ang.reindex(X_league.index)
        ang = ang.interpolate(method="linear", limit_direction="both")
        X = pd.concat([X_league, ang], axis=1)

    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return X


# ===========================================
# Build per-batter annual target + league exog
# ===========================================
def build_batter_annual(df: pd.DataFrame,
                        batter_id: str,
                        season_col: str = "season",
                        value_col: str = "exit_velo",
                        level_col: str = "level_abbr",
                        hit_type_col: str = "hit_type") -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      y: Series(float) indexed by annual PeriodIndex, mean exit_velo per season
      shares: DataFrame(float) same index, columns ['AA','AAA','MLB'] with league shares
      hit_type_shares: DataFrame(float) same index, columns for each hit type
    """
    sub = df.loc[df["batter_id"] == batter_id, [season_col, value_col, level_col, hit_type_col]].copy()
    if sub.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    sub[level_col] = sub[level_col].astype(str).str.strip().str.upper()
    sub[level_col] = sub[level_col].replace({
        "TRIPLE-A": "AAA", "DOUBLE-A": "AA", "MAJOR LEAGUE": "MLB", "ML": "MLB"
    })
    sub = sub[sub[level_col].isin(LEAGUES)]
    if sub.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    y = (sub.groupby(season_col)[value_col].mean().astype(float).sort_index())

    # One-hot encode hit_type and get seasonal shares
    hit_type_dummies = pd.get_dummies(sub[hit_type_col], prefix=hit_type_col, dtype=float)
    sub = pd.concat([sub, hit_type_dummies], axis=1)
    hit_type_shares = sub.groupby(season_col)[hit_type_dummies.columns].mean()

    # The rest of the function remains the same for y and league shares
    counts = (sub.assign(_cnt=1).groupby([season_col, level_col])["_cnt"].count().unstack(level_col, fill_value=0))
    for col in LEAGUES:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[LEAGUES]

    full_years = np.arange(int(y.index.min()), int(y.index.max()) + 1, dtype=int)
    y = y.reindex(full_years)
    counts = counts.reindex(full_years, fill_value=0)
    hit_type_shares = hit_type_shares.reindex(full_years, fill_value=0).interpolate(method="linear",
                                                                                    limit_direction="both")

    denom = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.divide(denom, axis=0)
    shares = shares.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both").fillna(method="ffill").fillna(method="bfill")

    y.index = pd.PeriodIndex(y.index.astype(int), freq="Y")
    shares.index = y.index
    hit_type_shares.index = y.index

    return y.astype(float), shares.astype(float), hit_type_shares.astype(float)


# ==========================
# Rolling validation (h=1)
# ==========================
def rolling_origin(y: pd.Series, X: pd.DataFrame, initial: int, horizon: int = 1):
    n = len(y)
    if n != len(X):
        raise ValueError("y and X must have same length.")
    if initial + horizon >= n:
        raise ValueError(f"Not enough annual points: initial={initial}, horizon={horizon}, n={n}")
    for t in range(initial, n - horizon):
        yield y.iloc[:t], X.iloc[:t, :], y.iloc[t:t + horizon], X.iloc[t:t + horizon, :]


def validate_auto_arimax(y: pd.Series, X: pd.DataFrame,
                         initial: int, horizon: int = 1) -> Dict[str, float]:
    preds, trues, aics = [], [], []
    order = (0, 0, 0)  # Default order
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
        except Exception:
            return {"rmse": np.inf, "mae": np.inf, "aic": np.inf, "order": None}
        preds.append(fc.values[-1]);
        trues.append(vy.iloc[-1])
        aics.append(res.aic() if np.isfinite(res.aic()) else np.inf)
        order = fit.order
    return {
        "rmse": rmse(trues, preds),
        "mae": float(np.mean(np.abs(np.array(trues) - np.array(preds)))),
        "aic": float(np.nanmean(aics)),
        "order": order
    }


def choose_arimax(y: pd.Series, X: pd.DataFrame,
                  min_initial_years: int = 5,
                  cv_horizon: int = 1) -> Dict[str, Any]:
    n = len(y)
    max_allowed_initial = n - cv_horizon - 1
    adaptive_initial = max(3, min(min_initial_years, max_allowed_initial)) if max_allowed_initial >= 3 else None

    if adaptive_initial is not None:
        scores = validate_auto_arimax(y, X, initial=adaptive_initial, horizon=cv_horizon)
        if scores["order"] is not None:
            return scores | {"mode": "cv", "initial_used": adaptive_initial, "horizon": cv_horizon}

    # AIC-only path
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
        return {"order": best_order, "rmse": np.nan, "mae": np.nan, "aic": aic,
                "mode": "aic", "initial_used": None, "horizon": cv_horizon}
    except Exception:
        return {"order": None, "rmse": np.nan, "mae": np.nan, "aic": np.inf,
                "mode": "fail", "initial_used": None, "horizon": cv_horizon}


# ===============================================
# Fit on full history + 2024 forecast (with exog)
# ===============================================
def build_exog_data(raw_df: pd.DataFrame,
                    batter_id: str,
                    season_col: str = "season",
                    la_col: str = "launch_angle",
                    spray_col: str = "spray_angle",
                    X_league: pd.DataFrame | None = None,
                    X_hit_type: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge season-level angle, league-share, and hit_type features.
    Returns X with all combined exogenous variables.
    """
    sub = raw_df.loc[raw_df["batter_id"] == batter_id].copy()
    ang = build_angle_features(sub, season_col=season_col,
                               la_col=la_col, spray_col=spray_col)

    exog_list = [df for df in [X_league, X_hit_type, ang] if df is not None and not df.empty]

    if not exog_list:
        return pd.DataFrame()

    first_df = exog_list[0]
    X = first_df.copy()
    for df in exog_list[1:]:
        X = pd.concat([X, df.reindex(X.index)], axis=1)

    X = X.interpolate(method="linear", limit_direction="both")
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    return X


def make_exog_2024_row(X_hist: pd.DataFrame,
                       target_level: str,
                       leagues: List[str] = LEAGUES) -> pd.DataFrame:
    """
    Build a 1-row exog for 2024:
      - set league one-hots to the requested target_level
      - use a linear projection for angle features from last 3 seasons
    """
    cols = X_hist.columns.tolist()
    last_row = X_hist.iloc[[-1]].copy()

    # Project angle features based on trend
    for col in [c for c in X_hist.columns if c not in leagues]:
        if len(X_hist) >= 3:
            trend = X_hist[col].iloc[-3:].diff().mean()
            last_row[col] = last_row[col].iloc[0] + trend

    for L in leagues:
        if L in last_row.columns:
            last_row[L] = 1.0 if L == target_level else 0.0

    next_idx = X_hist.index[-1] + 1
    last_row.index = [next_idx]
    return last_row[cols]


def refit_and_forecast_2024(y: pd.Series, X: pd.DataFrame, order: Tuple[int, int, int],
                            target_level: str) -> Dict[str, float]:
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

    # Add a simple uncertainty estimate based on historical differences
    if len(y) > 1:
        diffs = y.diff().dropna()
        if len(diffs) > 1:
            std_dev = diffs.std()
            lower = pred - 1.96 * std_dev
            upper = pred + 1.96 * std_dev
            return {"forecast": pred, "lower_95": lower, "upper_95": upper}

    return {"forecast": pred, "lower_95": np.nan, "upper_95": np.nan}


# ===========================================
# End-to-end per-batter 2024 with angle exog
# ===========================================
def predict_2024_per_batter(csv_path: str,
                            season_col: str = "season",
                            value_col: str = "exit_velo",
                            level_col: str = "level_abbr",
                            la_col: str = "launch_angle",
                            spray_col: str = "spray_angle",
                            target_level_mode: str = "last",
                            min_initial_years: int = 5,
                            output_csv: str = "per_batter_2024_arimax.csv") -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw = raw.dropna(subset=["batter_id", season_col, value_col, level_col]).copy()
    raw[season_col] = pd.to_numeric(raw[season_col], errors="coerce")
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna(subset=[season_col, value_col])

    raw[level_col] = raw[level_col].astype(str).str.strip().str.upper()
    raw[level_col] = raw[level_col].replace({
        "TRIPLE-A": "AAA", "DOUBLE-A": "AA", "MAJOR LEAGUE": "MLB", "ML": "MLB"
    })

    batters = raw["batter_id"].dropna().unique().tolist()
    out_rows: List[Dict[str, Any]] = []

    for i, b in enumerate(batters, 1):
        if i % 50 == 0:
            print(f"  ... {i}/{len(batters)}")

        y, shares, hit_type_shares = build_batter_annual(raw, b, season_col, value_col, level_col)
        if y.empty or shares.empty or hit_type_shares.empty or len(y) < 3:  # min history for ARMA
            continue

        X = build_exog_data(
            raw_df=raw,
            batter_id=b,
            season_col=season_col,
            la_col=la_col,
            spray_col=spray_col,
            X_league=shares,
            X_hit_type=hit_type_shares
        )

        last_year = int(y.index[-1].year)
        last_rows = raw[(raw["batter_id"] == b) & (raw[season_col] == last_year)]
        last_level = (last_rows[level_col].mode().iloc[0] if not last_rows.empty else "MLB")
        if last_level not in LEAGUES:
            last_level = "MLB"

        targets = [last_level] if target_level_mode == "last" else LEAGUES

        best = choose_arimax(
            y, X,
            min_initial_years=min_initial_years,
            cv_horizon=1
        )
        can_fit = (best["order"] is not None) and (
                np.isfinite(best.get("rmse", np.nan)) or best.get("mode") == "aic"
        )

        for L in targets:
            if can_fit:
                try:
                    pred = refit_and_forecast_2024(y, X, tuple(int(x) for x in best["order"]), L)
                    out_rows.append({
                        "batter_id": b,
                        "last_season": last_year,
                        "last_level": last_level,
                        "target_level_2024": L,
                        "method": f"ARIMAX{tuple(int(x) for x in best['order'])}",
                        "fit_mode": best.get("mode"),
                        "val_rmse": float(best.get("rmse", np.nan)),
                        "val_mae": float(best.get("mae", np.nan)),
                        "val_aic": float(best.get("aic", np.nan)),
                        **pred
                    })
                    continue
                except Exception as e:
                    print(f"Error for batter {b} with level {L}: {e}")
                    pass

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
    out.to_csv(output_csv, index=False)
    print(f"\nSaved 2024 per-batter forecasts to: {output_csv}")
    return out


# =========================
# Reporting & Visualization
# =========================
REPORT_DIR = Path("3_Modeling/time_series/outputs/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts(pred_csv: str, raw_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.read_csv(pred_csv)
    raw = pd.read_csv(raw_csv)
    return preds, raw


def summarize_kpis(preds: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    n_rows = len(preds)
    n_batters = preds["batter_id"].nunique()
    modeled_mask = preds["method"].str.startswith("ARIMAX", na=False)
    fallback_mask = ~modeled_mask
    is_last_scenario = preds["target_level_2024"] == preds["last_level"]

    subset = preds.loc[is_last_scenario].copy()
    modeled_last = subset[subset["method"].str.startswith("ARIMAX", na=False)]
    fallback_last = subset[~subset["method"].str.startswith("ARIMAX", na=False)]

    kpis = {
        "total_rows": n_rows,
        "unique_batters": n_batters,
        "rows_modeled": int(modeled_mask.sum()),
        "rows_fallback": int(fallback_mask.sum()),
        "pct_rows_modeled": float(100 * modeled_mask.mean()),
        "pct_rows_fallback": float(100 * fallback_mask.mean()),
        "batters_with_modeled_last_scenario": int(modeled_last["batter_id"].nunique()),
        "batters_with_fallback_last_scenario": int(fallback_last["batter_id"].nunique())
    }

    if not modeled_last.empty:
        rmse_vals = modeled_last["val_rmse"].replace([np.inf, -np.inf], np.nan).dropna()
        kpis.update({
            "cv_rmse_median": float(rmse_vals.median()) if not rmse_vals.empty else np.nan,
            "cv_rmse_mean": float(rmse_vals.mean()) if not rmse_vals.empty else np.nan,
            "cv_rmse_p75": float(rmse_vals.quantile(0.75)) if not rmse_vals.empty else np.nan,
            "modeled_rows_last_scenario": int(len(modeled_last)),
        })
    else:
        kpis.update({
            "cv_rmse_median": np.nan,
            "cv_rmse_mean": np.nan,
            "cv_rmse_p75": np.nan,
            "modeled_rows_last_scenario": 0
        })

    df_kpis = pd.DataFrame([kpis])
    df_kpis.to_csv(REPORT_DIR / "kpis_summary.csv", index=False)
    return df_kpis


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_method_mix(preds: pd.DataFrame, fname: str = "method_mix.png"):
    subset = preds[preds["target_level_2024"] == preds["last_level"]].copy()
    subset["is_modeled"] = subset["method"].str.startswith("ARIMAX", na=False)
    counts = subset["is_modeled"].value_counts().sort_index()
    labels = ["Fallback", "Modeled"] if len(counts) == 2 else ["Fallback", ]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.values)
    plt.title("Rows by Method (Last-Level Scenario)")
    plt.ylabel("Count")
    _savefig(REPORT_DIR / fname)


def plot_cv_rmse_hist(preds: pd.DataFrame, fname: str = "cv_rmse_hist.png"):
    subset = preds[(preds["target_level_2024"] == preds["last_level"]) &
                   (preds["method"].str.startswith("ARIMAX", na=False))].copy()
    vals = subset["val_rmse"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=20)
    plt.title("Cross-Validated RMSE (Modeled, Last-Level Scenario)")
    plt.xlabel("RMSE (mph)")
    plt.ylabel("Count")
    _savefig(REPORT_DIR / fname)


def attach_last_season_ev(preds: pd.DataFrame, raw: pd.DataFrame,
                          season_col: str = "season", value_col: str = "exit_velo") -> pd.DataFrame:
    hist = (raw.groupby(["batter_id", season_col])[value_col]
            .mean()
            .astype(float)
            .reset_index())
    last_year = (hist.groupby("batter_id")[season_col].max()
                 .reset_index().rename(columns={season_col: "last_year"}))
    last_ev = (hist.merge(last_year, on=["batter_id", ], how="inner"))
    last_ev = last_ev[last_ev[season_col] == last_ev["last_year"]][["batter_id", value_col]]
    last_ev = last_ev.rename(columns={value_col: "last_season_ev"})
    out = preds.merge(last_ev, on="batter_id", how="left")
    return out


def plot_forecast_vs_last(preds: pd.DataFrame, fname: str = "forecast_vs_last_scatter.png"):
    subset = preds[preds["target_level_2024"] == preds["last_level"]].copy()
    if "last_season_ev" not in subset.columns:
        return
    x = subset["last_season_ev"].astype(float)
    y = subset["forecast"].astype(float)
    colors = np.where(subset["method"].str.startswith("ARIMAX", na=False), "C0", "C1")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=15, alpha=0.7, c=colors)
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Last-Season EV (mph)")
    plt.ylabel("Forecast 2024 EV (mph)")
    plt.title("Forecast vs. Last-Season EV (Last-Level Scenario)")
    _savefig(REPORT_DIR / fname)


def plot_example_fit(raw: pd.DataFrame,
                     preds: pd.DataFrame,
                     batter_id: str,
                     season_col: str = "season",
                     value_col: str = "exit_velo",
                     fname_prefix: str = "example_fit_"):
    hist = (raw[raw["batter_id"] == batter_id]
            .groupby(season_col)[value_col].mean()
            .astype(float)
            .sort_index())
    row = (preds[(preds["batter_id"] == batter_id) &
                 (preds["target_level_2024"] == preds["last_level"])].head(1))
    if row.empty or hist.empty:
        return

    f = float(row["forecast"].iloc[0])
    lo = row["lower_95"].iloc[0]
    hi = row["upper_95"].iloc[0]
    last_year = int(hist.index.max())
    next_year = last_year + 1

    plt.figure(figsize=(7, 4))
    plt.plot(hist.index.values, hist.values, marker="o")
    plt.axvline(x=last_year, linestyle="--", alpha=0.5)

    # Plotting forecast point and CI
    if np.isfinite(lo) and np.isfinite(hi):
        plt.errorbar([next_year], [f], yerr=[[f - lo], [hi - f]], fmt="o", color="red", capsize=5)
    else:
        plt.plot([next_year], [f], marker="o", color="red")

    plt.title(f"Example Fit + Forecast: {batter_id}")
    plt.xlabel("Season")
    plt.ylabel("Mean Exit Velocity (mph)")
    _savefig(REPORT_DIR / f"{fname_prefix}{batter_id}.png")


def leaderboard_tables(preds: pd.DataFrame, top_k: int = 15):
    last = preds[preds["target_level_2024"] == preds["last_level"]].copy()
    top_proj = last.nlargest(top_k, "forecast")[["batter_id", "forecast", "lower_95", "upper_95", "method"]]
    top_proj.to_csv(REPORT_DIR / "table_top_projected.csv", index=False)

    modeled = last[last["method"].str.startswith("ARIMAX", na=False)].copy()
    if not modeled.empty:
        modeled["pi_width"] = modeled["upper_95"] - modeled["lower_95"]
        widest = modeled.nlargest(top_k, "pi_width")[["batter_id", "forecast", "lower_95", "upper_95", "pi_width"]]
        widest.to_csv(REPORT_DIR / "table_widest_intervals.csv", index=False)

        best_rmse = modeled.nsmallest(top_k, "val_rmse")[["batter_id", "val_rmse", "method"]]
        best_rmse.to_csv(REPORT_DIR / "table_best_cv_rmse.csv", index=False)


def make_report_assets(pred_csv: str,
                       raw_csv: str,
                       example_batters: List[str] | None = None):
    preds, raw = load_artifacts(pred_csv, raw_csv)
    kpis = summarize_kpis(preds)
    print("Saved KPIs:", REPORT_DIR / "kpis_summary.csv")
    print(kpis.to_string(index=False))

    preds_with_last = attach_last_season_ev(preds, raw)
    plot_method_mix(preds_with_last)
    plot_cv_rmse_hist(preds_with_last)
    plot_forecast_vs_last(preds_with_last)

    if example_batters is None:
        example_batters = preds_with_last["batter_id"].dropna().unique().tolist()[:5]
    for b in example_batters:
        plot_example_fit(raw, preds_with_last, b)

    leaderboard_tables(preds_with_last)
    print(f"Report assets written to: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    _ = predict_2024_per_batter(
        csv_path="1_Data/exit_velo_project_data.csv",
        season_col="season",
        value_col="exit_velo",
        level_col="level_abbr",
        la_col="launch_angle",
        spray_col="spray_angle",
        target_level_mode="all",
        min_initial_years=5,
        output_csv="3_Modeling/time_series/outputs/per_batter_2024_arimax.csv"
    )

    make_report_assets(
        pred_csv="3_Modeling/time_series/outputs/per_batter_2024_arimax.csv",
        raw_csv="1_Data/exit_velo_project_data.csv",
        example_batters=None
    )