"""
Quality-of-competition (QoC) extension: adjust each batter's estimated ability for
the strength of the pitchers they actually faced.

A full batted-ball-level model with crossed batter+pitcher random effects on 1.3M
rows is impractical for NUTS, so we use a tractable two-stage empirical-Bayes scheme:

  Stage 1 (pitcher effects).  Fit the base hierarchical model; for every batted ball
    compute the residual  r = exit_velo - E[EV | batter, level, age]  using posterior
    means. Average residuals by pitcher and SHRINK toward zero by sample size
    (James-Stein / empirical Bayes), giving each pitcher an EV-suppression effect
    phat_p (negative = suppresses exit velo).

  Stage 2 (QoC covariate).  For each (batter, season, level) cell, the quality of
    competition faced is QoC_c = mean phat over that cell's batted balls. Refit the
    hierarchical model with a term gamma * QoC_c, so the batter ability theta is
    estimated NET of the pitchers faced. Projection sets QoC to its mean (0) -> a
    competition-neutral ability.

The script runs a 2023 holdout (train <= 2022) comparing the base model vs the
QoC-adjusted model, since the only honest test of a QoC adjustment is whether it
improves out-of-sample accuracy.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pymc as pm

import hierarchical_model as hm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("qoc")

HOLDOUT = 2023
MIN_BBE = 50


# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_parquet(hm.DATA / "exit_velo_project_data.parquet")
    df["level_abbr"] = hm.normalize_levels(df["level_abbr"])
    df = df[df["level_abbr"].isin(hm.LEVELS)].copy()
    df["exit_velo"] = pd.to_numeric(df["exit_velo"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df.dropna(subset=["exit_velo", "batter_id", "season", "level_abbr",
                             "pitcher_id"])


def base_fit(cells, age_mean):
    model, batter_cat = hm.build_model(cells, age_mean)
    idata = hm.fit(model, draws=1000, tune=1500, chains=4)
    post = idata.posterior
    pm_ = lambda v: float(post[v].mean())
    theta = post["theta_batter"].mean(("chain", "draw")).values
    theta_map = dict(zip(batter_cat.categories, theta))
    params = {"mu0": pm_("mu_global"),
              "aaa": float(post["alpha_nonref"].mean(("chain", "draw")).values[0]),
              "aa": float(post["alpha_nonref"].mean(("chain", "draw")).values[1]),
              "b1": pm_("b_age1"), "b2": pm_("b_age2")}
    return idata, batter_cat, theta_map, params


def pitcher_effects(df: pd.DataFrame, theta_map, params, age_mean) -> pd.Series:
    """Stage 1: empirical-Bayes EV-suppression effect per pitcher (training data)."""
    a = (df["age"].to_numpy() - age_mean)
    lvl = df["level_abbr"].to_numpy()
    exp = (params["mu0"]
           + df["batter_id"].map(theta_map).fillna(0.0).to_numpy()
           + np.where(lvl == "AAA", params["aaa"], np.where(lvl == "AA", params["aa"], 0.0))
           + params["b1"] * a + params["b2"] * a ** 2)
    resid = df["exit_velo"].to_numpy() - exp

    g = pd.DataFrame({"pid": df["pitcher_id"].to_numpy(), "r": resid})
    agg = g.groupby("pid")["r"].agg(["mean", "size"])
    sigma2 = float(np.var(resid))                       # within (pooled) variance
    # between-pitcher variance via method of moments
    tau2 = max(1e-6, float(np.var(agg["mean"])) - sigma2 / agg["size"].mean())
    shrink = agg["size"] / (agg["size"] + sigma2 / tau2)   # -> 1 for big samples
    phat = agg["mean"] * shrink
    log.info("Pitcher effects: sigma2=%.2f tau2=%.3f (between-pitcher SD=%.2f mph); "
             "range [%.2f, %.2f]", sigma2, tau2, np.sqrt(tau2),
             phat.min(), phat.max())
    return phat                                          # index = pitcher_id


def cell_qoc(df: pd.DataFrame, phat: pd.Series) -> pd.DataFrame:
    """Stage 2 input: mean pitcher effect faced per (batter, season, level) cell."""
    df = df.assign(_p=df["pitcher_id"].map(phat).fillna(0.0))
    q = (df.groupby(["batter_id", "season", "level_abbr"])["_p"].mean()
           .rename("qoc").reset_index())
    return q


# ---------------------------------------------------------------------------
def build_qoc_model(cells, age_mean):
    """Base model + gamma * qoc covariate (qoc already merged onto cells)."""
    batter_cat = pd.Categorical(cells["batter_id"])
    bs_cat = pd.Categorical(cells["batter_id"].astype(str) + "_" + cells["season"].astype(str))
    age_c = cells["age"].to_numpy() - age_mean
    lc = cells["level_idx"].to_numpy()
    is_aaa = (lc == hm.LEVEL_IDX["AAA"]).astype(float)
    is_aa = (lc == hm.LEVEL_IDX["AA"]).astype(float)
    qoc = cells["qoc"].to_numpy()
    coords = {"batter": batter_cat.categories.astype(str),
              "batter_season": bs_cat.categories.astype(str),
              "level_nonref": ["AAA", "AA"]}
    with pm.Model(coords=coords) as model:
        mu0 = pm.Normal("mu_global", 89.0, 5.0)
        alpha = pm.Normal("alpha_nonref", mu=np.array([-1.5, -1.5]), sigma=4.0,
                          dims="level_nonref")
        b1 = pm.Normal("b_age1", 0.0, 0.5)
        b2 = pm.Normal("b_age2", 0.0, 0.1)
        gamma = pm.Normal("gamma_qoc", 0.0, 1.0)         # QoC coefficient (expect ~1)
        s_b = pm.HalfNormal("sigma_batter", 5.0)
        z_b = pm.Normal("z_batter", 0.0, 1.0, dims="batter")
        theta = pm.Deterministic("theta_batter", z_b * s_b, dims="batter")
        s_s = pm.HalfNormal("sigma_season", 2.0)
        z_s = pm.Normal("z_bs", 0.0, 1.0, dims="batter_season")
        mu = (mu0 + theta[batter_cat.codes] + z_s[bs_cat.codes] * s_s
              + alpha[0] * is_aaa + alpha[1] * is_aa
              + b1 * age_c + b2 * age_c ** 2
              + gamma * qoc)
        pm.Normal("obs", mu=mu, sigma=cells["se"].to_numpy(),
                  observed=cells["mean_ev"].to_numpy())
    return model, batter_cat


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    df = load()
    train = df[df.season <= HOLDOUT - 1].copy()
    age_mean = float(train["age"].mean())

    # ground truth for 2023
    mlb = df[(df.season == HOLDOUT) & (df.level_abbr == "MLB")]
    g = mlb.groupby("batter_id").agg(truth=("exit_velo", "mean"),
                                     n=("exit_velo", "size"),
                                     age=("age", "mean")).reset_index()
    g = g[(g.n >= MIN_BBE) & (g.batter_id.isin(set(train.batter_id)))].reset_index(drop=True)
    log.info("Holdout batters: %d", len(g))

    # ----- base model (no QoC) -----
    cells, _ = hm.build_cells(train)
    log.info("Fitting BASE model ...")
    _, bcat, theta_map, params = base_fit(cells, age_mean)
    base_pred = hm.project_2024(_, bcat, g[["batter_id", "age"]], age_mean) \
        .set_index("batter_id")["exit_velo_pred"]
    g["base"] = g.batter_id.map(base_pred)

    # ----- QoC model -----
    phat = pitcher_effects(train, theta_map, params, age_mean)
    qoc = cell_qoc(train, phat)
    cells_q = cells.merge(qoc, on=["batter_id", "season", "level_abbr"], how="left")
    cells_q["qoc"] = cells_q["qoc"].fillna(0.0)
    log.info("Cell QoC: SD=%.3f mph (how much avg competition varies across cells)",
             cells_q["qoc"].std())

    log.info("Fitting QoC-adjusted model ...")
    qmodel, qcat = build_qoc_model(cells_q, age_mean)
    qidata = hm.fit(qmodel, draws=1000, tune=1500, chains=4)
    gamma = float(qidata.posterior["gamma_qoc"].mean())
    gamma_hdi = np.percentile(qidata.posterior["gamma_qoc"].values, [3, 97])
    log.info("gamma_qoc = %.2f  [%.2f, %.2f]", gamma, *gamma_hdi)
    qoc_pred = hm.project_2024(qidata, qcat, g[["batter_id", "age"]], age_mean) \
        .set_index("batter_id")["exit_velo_pred"]
    g["qoc_adj"] = g.batter_id.map(qoc_pred)

    res = pd.DataFrame({
        "method": ["base_hierarchical", "qoc_adjusted"],
        "rmse": [rmse(g.truth, g.base), rmse(g.truth, g.qoc_adj)],
        "mae": [float(np.mean(np.abs(g.truth - g.base))),
                float(np.mean(np.abs(g.truth - g.qoc_adj)))],
        "corr": [float(np.corrcoef(g.truth, g.base)[0, 1]),
                 float(np.corrcoef(g.truth, g.qoc_adj)[0, 1])],
    })
    res.to_csv(hm.OUT / "qoc_backtest_metrics.csv", index=False)

    # pitcher leaderboard (most/least EV-suppressing, min 200 BBE)
    counts = train.groupby("pitcher_id").size()
    board = phat[counts[counts >= 200].index.intersection(phat.index)]
    board.sort_values().to_csv(hm.OUT / "pitcher_effects.csv", header=["ev_effect_mph"])

    log.info("\n=== QoC vs BASE (2023 holdout, n=%d) ===\n%s\n"
             "gamma_qoc=%.2f [%.2f,%.2f]  cell-QoC SD=%.3f mph",
             len(g), res.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
             gamma, gamma_hdi[0], gamma_hdi[1], cells_q["qoc"].std())


if __name__ == "__main__":
    main()
