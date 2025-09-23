import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_SIMULATIONS     = 1000
MATCHES_PER_SIM   = 3000
K_FACTOR          = 32
INITIAL_RATING    = 1000
RNG_SEED          = 123
# =========================
# Utilities
# =========================
rng = np.random.default_rng(RNG_SEED)

def _to_lower(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_events(path):
    df = pd.read_csv(path)
    df = _to_lower(df)
    required = ["batter_id", "season", "exit_velo", "age", "hit_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # keep only MLB if present
    if "level_abbr" in df.columns:
        df = df[df["level_abbr"].str.lower() == "mlb"]
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season", "exit_velo", "age"])
    df["batter_id"] = df["batter_id"].astype(str)
    df["hit_type"] = df["hit_type"].astype(str)
    return df

def choose_season(df, season=None):
    if season is None:
        season = int(df["season"].dropna().max())
    out = df[df["season"] == season].copy()
    if out.empty:
        raise ValueError(f"No rows found for season={season}")
    return out, season

def zscore(x):
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    sd = 1.0 if (not np.isfinite(sd) or sd == 0) else sd
    return (x - mu) / sd

def build_event_scores(df_season, cfg):
    """Return dict: player -> np.array of event scores, and a players list in fixed order."""
    df = df_season.copy()

    # z-exit velo within this season
    df["ev_z"] = zscore(df["exit_velo"].values)

    # age z across players in this season
    age_z_series = df.groupby("batter_id")["age"].transform("mean")  # player-level age
    df["age_z"] = zscore(age_z_series.values) if cfg["use_age"] else 0.0

    # hit_type mapping
    if cfg["use_hit_type"]:
        hmap = cfg["hit_type_map"]
        df["ht_score"] = df["hit_type"].map(hmap).fillna(0.0)
    else:
        df["ht_score"] = 0.0

    # weighted event score
    df["event_score"] = (
        df["ev_z"]
        + cfg["age_weight"] * df["age_z"]
        + cfg["hit_type_weight"] * df["ht_score"]
    )

    # filter players with too few events
    counts = df.groupby("batter_id").size()
    keep_players = counts[counts >= cfg["min_events_per_player"]].index.tolist()
    df = df[df["batter_id"].isin(keep_players)].copy()

    players = sorted(df["batter_id"].unique().tolist())

    # build arrays per player
    events_by_player = {}
    for p in players:
        events_by_player[p] = df.loc[df["batter_id"] == p, "event_score"].to_numpy(dtype=float)

    return players, events_by_player

def run_elo_sim(players, events_by_player, n_sims, matches_per_sim, k, init_rating, seed=123):
    rng = np.random.default_rng(seed)
    n_players = len(players)
    if n_players < 2:
        raise ValueError("Need at least two players to compute Elo.")
    ratings_all = np.zeros((n_sims, n_players), dtype=float)

    # quick index for vectorized access
    player_index = {p: i for i, p in enumerate(players)}

    for s in range(n_sims):
        ratings = np.full(n_players, init_rating, dtype=float)

        # random pairings with replacement
        p_idx1 = rng.integers(0, n_players, size=matches_per_sim)
        p_idx2 = rng.integers(0, n_players, size=matches_per_sim)

        # ensure p1 != p2 (re-draw where equal)
        same = p_idx1 == p_idx2
        if np.any(same):
            p_idx2[same] = (p_idx2[same] + 1) % n_players

        for i1, i2 in zip(p_idx1, p_idx2):
            p1 = players[i1]; p2 = players[i2]
            ev1 = events_by_player[p1]; ev2 = events_by_player[p2]
            if ev1.size == 0 or ev2.size == 0:
                continue

            # sample one event per player
            s1 = ev1[rng.integers(0, ev1.size)]
            s2 = ev2[rng.integers(0, ev2.size)]

            # outcome
            if s1 > s2:
                outcome = 1.0
            elif s1 < s2:
                outcome = 0.0
            else:
                outcome = 0.5

            # expected score from Elo
            expected = 1.0 / (1.0 + 10.0 ** ((ratings[i2] - ratings[i1]) / 400.0))
            delta = k * (outcome - expected)
            ratings[i1] += delta
            ratings[i2] -= delta

        ratings_all[s, :] = ratings

    return ratings_all

def summarize_elo(players, ratings_all, init_rating):
    mean_elo = ratings_all.mean(axis=0)
    sd_elo   = ratings_all.std(axis=0, ddof=1)
    ci_l     = np.quantile(ratings_all, 0.025, axis=0)
    ci_u     = np.quantile(ratings_all, 0.975, axis=0)

    # relative_strength and ranking_confidence as in your R
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_strength = (mean_elo - init_rating) / np.where(sd_elo == 0, np.nan, sd_elo)
        rank_conf    = (ci_u - ci_l) / np.where(mean_elo == 0, np.nan, mean_elo)

    df = pd.DataFrame({
        "batter_id": players,
        "mean_elo": mean_elo,
        "ci_lower": ci_l,
        "ci_upper": ci_u,
        "sd_elo": sd_elo,
        "relative_strength": rel_strength,
        "ranking_confidence": rank_conf
    }).sort_values("mean_elo", ascending=False).reset_index(drop=True)

    # quartile buckets like your R code
    df["priority_level"] = pd.qcut(
        df["mean_elo"],
        q=4,
        labels=["Low", "Medium", "High", "Top Priority"]
    )
    return df

def plot_elo(df, season, init_rating, out_path_pdf):
    plt.figure(figsize=(10, max(6, 0.3*len(df))))
    df_plot = df.sort_values("mean_elo").reset_index(drop=True)
    y = np.arange(len(df_plot))
    plt.axvline(x=init_rating, linestyle="--", color="gray", alpha=0.6)
    # error bars
    xerr = np.vstack([
        df_plot["mean_elo"] - df_plot["ci_lower"],
        df_plot["ci_upper"] - df_plot["mean_elo"]
    ])
    plt.errorbar(df_plot["mean_elo"], y, xerr=xerr, fmt="o", capsize=3)
    plt.yticks(y, df_plot["batter_id"])
    plt.xlabel("Elo rating")
    title_season = f" (season {season})" if season is not None else ""
    plt.title(f"Player Elo based on event-level comparisons{title_season}")
    plt.tight_layout()
    plt.savefig(out_path_pdf)
    plt.close()
