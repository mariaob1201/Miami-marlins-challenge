"""
Hierarchical Bayesian model for projecting underlying MLB exit-velocity ability.

WHY THIS MODEL (vs. the per-batter ARIMA in 3_Modeling/time_series/)
--------------------------------------------------------------------
The task is to estimate each hitter's *underlying* average exit velocity against
MLB-level competition and project it into 2024. Two facts about the data make a
hierarchical model the right tool:

  1. Sparsity. Most batters have very few seasons (1,642 of 3,715 have a single
     season) and wildly different sample sizes per season (1 to ~700 batted
     balls). A noisy 12-BBE estimate must be trusted *less* than a 600-BBE one.
  2. Cross-level projection. ~100 of the 647 target batters have only
     minor-league history, so projecting their MLB ability requires an estimated
     level adjustment (MLB <- AAA <- AA), not a hard-coded constant.

A partial-pooling (hierarchical) model addresses both directly:
  * Each batter's ability is shrunk toward the population mean by an amount that
    depends on how much (and how precise) their data is  ->  regression to the
    mean falls out automatically.
  * Level equivalencies (alpha_level) and an age curve are *estimated* from
    players observed across levels/ages, with full posterior uncertainty.

EFFICIENT LIKELIHOOD (measurement-error formulation)
----------------------------------------------------
Rather than modeling all 1.3M batted balls directly (slow, and the per-ball
scatter is nuisance noise), we collapse to (batter, season, level) cells. For a
cell with n batted balls and sample SD s, the cell mean has a known standard
error se = s / sqrt(n). We model the cell mean as

    cell_mean ~ Normal(mu_cell, sqrt(se^2 + sigma_season^2))

where sigma_season captures genuine year-to-year ability fluctuation beyond
sampling noise. This is a standard meta-analytic / measurement-error model: it
reduces ~1.3M rows to ~6k cells while preserving exactly the sample-size
weighting we want.

    mu_cell = mu_global + theta_batter[b]              # latent ability (pooled)
                        + alpha_level[level]           # MLB=0 reference
                        + b_age1 * age_c + b_age2 * age_c^2

Projection: a batter's 2024 MLB ability is
    mu_global + theta_batter[b] + alpha_level[MLB=0] + age curve at 2024 age.
Batters with no history fall back to the population mean (theta=0) with wide
intervals that include the between-batter SD.

Author: Maria Oros  (hardened / promoted from the deprecated PyMC draft)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("hierarchical")

RANDOM_SEED = 40
rng = np.random.default_rng(RANDOM_SEED)

# Repo root is two levels up from this file (3_Modeling/hierarchical/ -> repo).
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "1_Data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = ["MLB", "AAA", "AA"]            # index 0,1,2 ; MLB is the reference
LEVEL_IDX = {lvl: i for i, lvl in enumerate(LEVELS)}
TARGET_SEASON = 2024


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def normalize_levels(s: pd.Series) -> pd.Series:
    """Map the many spellings of a level to {MLB, AAA, AA}."""
    return (s.astype(str).str.strip().str.upper()
            .replace({"TRIPLE-A": "AAA", "AAA BALL": "AAA",
                      "DOUBLE-A": "AA",
                      "MAJOR LEAGUE": "MLB", "ML": "MLB"}))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the batted-ball parquet and the 2024 validation roster."""
    pq = DATA / "exit_velo_project_data.parquet"
    log.info("Loading %s", pq.name)
    df = pd.read_parquet(pq)
    df["level_abbr"] = normalize_levels(df["level_abbr"])
    df = df[df["level_abbr"].isin(LEVELS)].copy()
    df["exit_velo"] = pd.to_numeric(df["exit_velo"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["exit_velo", "batter_id", "season", "level_abbr"])

    valid = pd.read_csv(DATA / "exit_velo_validate_data.csv")
    return df, valid


def build_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Collapse batted balls to (batter, season, level) cells with the mean exit
    velocity, the sample size, and the standard error of the mean.

    For single-BBE cells (no within-cell SD) we impute the pooled within-cell SD
    so their standard error is large -> they get shrunk heavily.
    """
    g = (df.groupby(["batter_id", "season", "level_abbr"])
           .agg(mean_ev=("exit_velo", "mean"),
                sd_ev=("exit_velo", "std"),
                n=("exit_velo", "size"),
                age=("age", "mean"))
           .reset_index())

    pooled_sd = float(df["exit_velo"].std())            # ~14.5 mph
    g["sd_ev"] = g["sd_ev"].fillna(pooled_sd)
    # n==1 cells get sd_ev=pooled; clip tiny SDs so se never collapses to 0.
    g["sd_ev"] = g["sd_ev"].clip(lower=1.0)
    g["se"] = g["sd_ev"] / np.sqrt(g["n"])

    g["level_idx"] = g["level_abbr"].map(LEVEL_IDX).astype(int)
    log.info("Built %d batter-season-level cells from %d batted balls (%d batters)",
             len(g), len(df), g["batter_id"].nunique())
    return g, pooled_sd


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(cells: pd.DataFrame, age_mean: float) -> tuple[pm.Model, pd.CategoricalDtype]:
    """
    Construct the hierarchical measurement-error model.

    Identification note: the level effect (alpha_level) is identified from cells
    that share the same batter AND season but differ in level. We therefore add a
    batter-season "form" random intercept (eta). Without it, alpha would soak up
    across-season development/selection differences and even flip sign -- the raw
    cross-season contrast and the clean within-batter-season contrast point in
    OPPOSITE directions in this data, so this structure is essential.
    """
    batter_cat = pd.Categorical(cells["batter_id"])

    # batter-season grouping for the "form" random intercept.
    bs_key = cells["batter_id"].astype(str) + "_" + cells["season"].astype(str)
    bs_cat = pd.Categorical(bs_key)

    age_c = (cells["age"].to_numpy() - age_mean)
    level_codes = cells["level_idx"].to_numpy()
    # level design for the two NON-reference levels (AAA, AA); MLB is baseline 0.
    is_aaa = (level_codes == LEVEL_IDX["AAA"]).astype(float)
    is_aa = (level_codes == LEVEL_IDX["AA"]).astype(float)

    coords = {"batter": batter_cat.categories.astype(str),
              "batter_season": bs_cat.categories.astype(str),
              "level_nonref": ["AAA", "AA"]}

    with pm.Model(coords=coords) as model:
        mu_global = pm.Normal("mu_global", mu=89.0, sigma=5.0)

        # Level equivalencies (deltas vs MLB). Prior centered near the clean
        # within-batter-season contrast (~ -1.75) but WIDE, so data drives it.
        alpha_nonref = pm.Normal("alpha_nonref", mu=np.array([-1.5, -1.5]),
                                 sigma=4.0, dims="level_nonref")
        alpha_aaa = alpha_nonref[0]
        alpha_aa = alpha_nonref[1]

        # Age curve (centered age). Mild priors: EV tends to peak in mid/late 20s.
        b_age1 = pm.Normal("b_age1", mu=0.0, sigma=0.5)
        b_age2 = pm.Normal("b_age2", mu=0.0, sigma=0.1)

        # Between-batter ability spread + non-centered batter effects.
        sigma_batter = pm.HalfNormal("sigma_batter", sigma=5.0)
        z_batter = pm.Normal("z_batter", mu=0.0, sigma=1.0, dims="batter")
        theta_batter = pm.Deterministic("theta_batter", z_batter * sigma_batter,
                                        dims="batter")

        # Batter-season "form" intercept: shared across a player's same-season
        # cells, so the level coefficient is identified WITHIN batter-season.
        sigma_season = pm.HalfNormal("sigma_season", sigma=2.0)
        z_bs = pm.Normal("z_bs", mu=0.0, sigma=1.0, dims="batter_season")
        eta_bs = z_bs * sigma_season

        mu_cell = (mu_global
                   + theta_batter[batter_cat.codes]
                   + eta_bs[bs_cat.codes]
                   + alpha_aaa * is_aaa
                   + alpha_aa * is_aa
                   + b_age1 * age_c
                   + b_age2 * age_c ** 2)

        # Likelihood noise is the (known) sampling error of each cell mean; the
        # season "form" variation is now modeled explicitly by eta_bs.
        se = cells["se"].to_numpy()
        pm.Normal("obs", mu=mu_cell, sigma=se,
                  observed=cells["mean_ev"].to_numpy())

    return model, batter_cat


def fit(model: pm.Model, draws: int = 1500, tune: int = 2000, chains: int = 4):
    log.info("Sampling: %d draws x %d chains (tune=%d)", draws, chains, tune)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.95, random_seed=RANDOM_SEED,
                          progressbar=True)
    n_div = int(idata.sample_stats["diverging"].sum())
    log.info("Divergences: %d", n_div)
    return idata


# ---------------------------------------------------------------------------
# Projection to 2024 (MLB ability)
# ---------------------------------------------------------------------------
def project_2024(idata, batter_cat, valid: pd.DataFrame, age_mean: float) -> pd.DataFrame:
    """
    Posterior of each target batter's 2024 MLB exit-velocity ability:
        mu_global + theta_batter[b] + alpha_level[MLB=0] + age curve(2024 age)
    Cold-start batters (no history) use theta ~ Normal(0, sigma_batter).
    """
    post = idata.posterior
    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    flat = lambda v: post[v].values.reshape(n_chain * n_draw, -1).squeeze()

    mu_global = flat("mu_global")                  # (S,)
    b_age1 = flat("b_age1")
    b_age2 = flat("b_age2")
    sigma_batter = flat("sigma_batter")
    theta = post["theta_batter"].values.reshape(n_chain * n_draw, -1)  # (S, n_batter)
    cat_to_col = {c: i for i, c in enumerate(batter_cat.categories)}
    S = mu_global.shape[0]

    rows = []
    for _, r in valid.iterrows():
        bid = r["batter_id"]
        age_c = float(r["age"]) - age_mean
        if bid in cat_to_col:
            theta_b = theta[:, cat_to_col[bid]]
            seen = True
        else:                                       # cold start -> draw from population
            theta_b = rng.normal(0.0, sigma_batter)
            seen = False
        ability = (mu_global + theta_b              # MLB reference: alpha_level = 0
                   + b_age1 * age_c + b_age2 * age_c ** 2)
        lo, hi = np.percentile(ability, [2.5, 97.5])
        rows.append({
            "season": TARGET_SEASON,
            "batter_id": bid,
            "age": r["age"],
            "exit_velo_pred": float(ability.mean()),
            "lower_95": float(lo),
            "upper_95": float(hi),
            "pred_std": float(ability.std()),
            "has_history": seen,
        })
    out = pd.DataFrame(rows).sort_values("batter_id").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_reports(idata, preds: pd.DataFrame):
    preds.to_csv(OUT / "per_batter_2024_predictions.csv", index=False)

    summ = az.summary(idata, var_names=["mu_global", "alpha_nonref", "b_age1",
                                        "b_age2", "sigma_batter", "sigma_season"],
                      round_to=3)
    summ.to_csv(OUT / "parameter_summary.csv")

    max_rhat = float(az.rhat(idata).to_array().max())
    lines = [
        "HIERARCHICAL EXIT-VELOCITY MODEL  -  2024 MLB ABILITY PROJECTIONS",
        "=" * 64,
        f"Target batters projected : {len(preds):,}",
        f"  with history           : {int(preds['has_history'].sum()):,}",
        f"  cold-start (pooled)    : {int((~preds['has_history']).sum()):,}",
        "",
        f"Projected EV (mph): mean {preds['exit_velo_pred'].mean():.2f}, "
        f"range [{preds['exit_velo_pred'].min():.1f}, {preds['exit_velo_pred'].max():.1f}]",
        f"Mean 95% interval width: "
        f"{(preds['upper_95'] - preds['lower_95']).mean():.2f} mph",
        "",
        "Estimated level equivalencies (mph vs MLB):",
        f"  AAA: {summ.loc['alpha_nonref[AAA]','mean']:+.2f} "
        f"[{summ.loc['alpha_nonref[AAA]','hdi_3%']:+.2f}, {summ.loc['alpha_nonref[AAA]','hdi_97%']:+.2f}]",
        f"  AA : {summ.loc['alpha_nonref[AA]','mean']:+.2f} "
        f"[{summ.loc['alpha_nonref[AA]','hdi_3%']:+.2f}, {summ.loc['alpha_nonref[AA]','hdi_97%']:+.2f}]",
        "",
        f"Between-batter SD (sigma_batter): {summ.loc['sigma_batter','mean']:.2f} mph",
        f"Season noise   (sigma_season)  : {summ.loc['sigma_season','mean']:.2f} mph",
        f"Age curve: linear {summ.loc['b_age1','mean']:+.3f}, quad {summ.loc['b_age2','mean']:+.4f}",
        "",
        f"Convergence: max R-hat = {max_rhat:.3f} "
        f"({'OK' if max_rhat < 1.01 else 'CHECK'})",
    ]
    (OUT / "summary_report.txt").write_text("\n".join(lines))
    log.info("\n%s", "\n".join(lines))


def main():
    df, valid = load_data()
    cells, _ = build_cells(df)
    age_mean = float(df["age"].mean())
    log.info("Age centered at %.2f", age_mean)

    model, batter_cat = build_model(cells, age_mean)
    idata = fit(model)
    preds = project_2024(idata, batter_cat, valid, age_mean)
    write_reports(idata, preds)
    # NOTE: the full InferenceData (group-level draws for ~6k batter-seasons) is
    # ~200 MB and exceeds GitHub limits, so we do NOT persist it. The predictions
    # CSV + parameter_summary.csv fully reproduce the reported results; set
    # SAVE_TRACE=1 to keep a local copy for deeper diagnostics.
    if os.environ.get("SAVE_TRACE") == "1":
        idata.to_netcdf(str(OUT / "idata.nc"))
        log.info("Saved full trace to idata.nc (local only, git-ignored)")
    log.info("Done. Outputs in %s", OUT)


if __name__ == "__main__":
    main()
