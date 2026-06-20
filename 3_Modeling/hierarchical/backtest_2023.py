"""
Backtest: hold out 2023 and score competing projection methods against observed
2023 MLB exit velocity.

Protocol
--------
* TRAIN on seasons <= 2022 only (strict temporal holdout -- no leakage).
* GROUND TRUTH = each batter's observed 2023 MLB mean exit velo, restricted to
  batters with >= MIN_BBE MLB batted balls in 2023 so the target itself is stable
  (a 5-BBE "truth" is mostly sampling noise no model can predict).
* Score every method on the SAME batter set (those with both 2023 truth and
  pre-2023 history) so the comparison is apples-to-apples.

Methods compared
----------------
  hierarchical   : the PyMC partial-pooling model (this project's primary model),
                   trained on <=2022 and projected to 2023 MLB at 2023 age.
  arima_ts       : per-batter annual ARIMA on a level-adjusted MLB-equivalent
                   series (statsmodels), mirroring 3_Modeling/time_series/arima.py.
  last_mlb       : most recent MLB-season mean (fallback: last level + level adj).
  career_mlb     : career MLB mean (fallback: all-level mean + level adj).
  global_mean    : constant population mean (the dumbest baseline).

Metrics: RMSE, MAE, bias (mean error), and Pearson r vs truth.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import hierarchical_model as hm   # same directory

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backtest")

HOLDOUT = 2023
MIN_BBE = 50            # min 2023 MLB batted balls to count as stable ground truth


# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_parquet(hm.DATA / "exit_velo_project_data.parquet")
    df["level_abbr"] = hm.normalize_levels(df["level_abbr"])
    df = df[df["level_abbr"].isin(hm.LEVELS)].copy()
    df["exit_velo"] = pd.to_numeric(df["exit_velo"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df.dropna(subset=["exit_velo", "batter_id", "season", "level_abbr"])


def level_offsets(train: pd.DataFrame) -> dict[str, float]:
    """
    Empirical MLB-equivalent offsets from WITHIN-batter-season level contrasts on
    training data: offset[L] = mean(level_L_cell - MLB_cell) over batter-seasons
    seen at both. MLB-equiv value = raw - offset[L].
    """
    cell = (train.groupby(["batter_id", "season", "level_abbr"])["exit_velo"]
            .mean().reset_index())
    piv = cell.pivot_table(index=["batter_id", "season"],
                           columns="level_abbr", values="exit_velo")
    off = {"MLB": 0.0}
    for L in ("AAA", "AA"):
        if L in piv and "MLB" in piv:
            off[L] = float((piv[L] - piv["MLB"]).dropna().mean())
        else:
            off[L] = 0.0
    log.info("Level offsets (mph above MLB): %s",
             {k: round(v, 2) for k, v in off.items()})
    return off


# ---------------------------------------------------------------------------
# Ground truth + per-batter training views
# ---------------------------------------------------------------------------
def ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    mlb = df[(df.season == HOLDOUT) & (df.level_abbr == "MLB")]
    g = mlb.groupby("batter_id").agg(truth=("exit_velo", "mean"),
                                     n_bbe=("exit_velo", "size"),
                                     age=("age", "mean")).reset_index()
    return g[g.n_bbe >= MIN_BBE].copy()


def mlb_equiv_annual(train_b: pd.DataFrame, off: dict[str, float]) -> pd.Series:
    """Annual MLB-equivalent mean EV series for one batter (training only)."""
    train_b = train_b.copy()
    train_b["adj"] = train_b["exit_velo"] - train_b["level_abbr"].map(off)
    return train_b.groupby("season")["adj"].mean().sort_index()


# ---------------------------------------------------------------------------
# Baseline predictors (all use ONLY training data <=2022)
# ---------------------------------------------------------------------------
def pred_last_mlb(train_b, off):
    mlb = train_b[train_b.level_abbr == "MLB"]
    if not mlb.empty:
        last = int(mlb.season.max())
        return float(mlb[mlb.season == last].exit_velo.mean())
    # fallback: most recent level, converted to MLB-equiv
    last = int(train_b.season.max())
    sub = train_b[train_b.season == last]
    return float((sub.exit_velo - sub.level_abbr.map(off)).mean())


def pred_career_mlb(train_b, off):
    mlb = train_b[train_b.level_abbr == "MLB"]
    if not mlb.empty:
        return float(mlb.exit_velo.mean())
    return float((train_b.exit_velo - train_b.level_abbr.map(off)).mean())


def pred_arima_ts(series: pd.Series):
    """
    Per-batter ARIMA on the annual MLB-equiv series, faithful to arima.py's spirit:
    pick the lowest-AIC small order when enough points exist, else random-walk /
    last value. Series index is calendar year.
    """
    y = series.dropna().astype(float)
    if len(y) == 0:
        return np.nan
    if len(y) < 4:                      # too short for ARIMA -> persistence
        return float(y.iloc[-1])
    yv = y.reset_index(drop=True)
    best_aic, best_fc = np.inf, float(yv.iloc[-1])
    for order in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1)]:
        try:
            res = ARIMA(yv, order=order, trend="t" if order[1] == 0 else "n",
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit()
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_fc = float(res.get_forecast(1).predicted_mean.iloc[0])
        except Exception:
            continue
    return best_fc


# ---------------------------------------------------------------------------
def run_hierarchical(train: pd.DataFrame, roster: pd.DataFrame) -> pd.Series:
    """Train the PyMC model on <=2022 and project 2023 MLB ability per batter."""
    cells, _ = hm.build_cells(train)
    age_mean = float(train["age"].mean())
    model, batter_cat = hm.build_model(cells, age_mean)
    idata = hm.fit(model, draws=1000, tune=1500, chains=4)
    roster_in = roster.rename(columns={"age": "age"})[["batter_id", "age"]]
    preds = hm.project_2024(idata, batter_cat, roster_in, age_mean)
    return preds.set_index("batter_id")["exit_velo_pred"]


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    e = pred - truth
    return {"rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "corr": float(np.corrcoef(truth, pred)[0, 1])}


def main():
    df = load()
    train = df[df.season <= HOLDOUT - 1].copy()
    off = level_offsets(train)

    truth = ground_truth(df)
    have_hist = set(train.batter_id)
    truth = truth[truth.batter_id.isin(have_hist)].reset_index(drop=True)
    log.info("Backtest set: %d batters (>= %d MLB BBE in %d, with pre-%d history)",
             len(truth), MIN_BBE, HOLDOUT, HOLDOUT)

    # per-batter baselines
    preds = {m: [] for m in ["arima_ts", "last_mlb", "career_mlb"]}
    gtrain = {b: g for b, g in train.groupby("batter_id")}
    for bid in truth.batter_id:
        tb = gtrain[bid]
        preds["last_mlb"].append(pred_last_mlb(tb, off))
        preds["career_mlb"].append(pred_career_mlb(tb, off))
        preds["arima_ts"].append(pred_arima_ts(mlb_equiv_annual(tb, off)))
    for m in preds:
        truth[m] = preds[m]
    truth["global_mean"] = float(train[train.level_abbr == "MLB"].exit_velo.mean())

    # hierarchical (single fit, projects all)
    log.info("Training hierarchical model on <=%d ...", HOLDOUT - 1)
    hier = run_hierarchical(train, truth[["batter_id", "age"]])
    truth["hierarchical"] = truth.batter_id.map(hier)

    methods = ["hierarchical", "arima_ts", "last_mlb", "career_mlb", "global_mean"]
    rows = []
    y = truth["truth"].to_numpy()
    for m in methods:
        p = truth[m].to_numpy()
        mask = np.isfinite(p)
        rows.append({"method": m, "n": int(mask.sum()), **metrics(y[mask], p[mask])})
    res = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    out = hm.OUT / f"backtest_{HOLDOUT}_metrics.csv"
    res.to_csv(out, index=False)
    truth.to_csv(hm.OUT / f"backtest_{HOLDOUT}_predictions.csv", index=False)

    log.info("\n=== %d HOLDOUT BACKTEST (n=%d, target = observed MLB mean, "
             ">=%d BBE) ===\n%s", HOLDOUT, len(truth), MIN_BBE,
             res.to_string(index=False,
                           float_format=lambda v: f"{v:.3f}"))
    best = res.iloc[0]["method"]
    naive = res[res.method == "career_mlb"].iloc[0]["rmse"]
    hr = res[res.method == "hierarchical"].iloc[0]["rmse"]
    log.info("Hierarchical RMSE %.3f vs career_mlb %.3f  (%.1f%% lower)",
             hr, naive, 100 * (naive - hr) / naive)
    log.info("Best method: %s", best)


if __name__ == "__main__":
    main()
