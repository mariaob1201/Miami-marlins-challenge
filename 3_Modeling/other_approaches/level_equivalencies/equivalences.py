# Level equivalencies
# I used ELO score in the past, I will be using it here

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import *
# =========================
# CONFIG
# =========================
INPUT_CSV = "1_Data/exit_velo_project_data.csv"
OUT_DIR   = "out_elo"
SEASON    = None

# Elo & simulation settings
N_SIMULATIONS     = 1000
MATCHES_PER_SIM   = 3000
K_FACTOR          = 32
INITIAL_RATING    = 1000
RNG_SEED          = 123


CONFIG = {
    "use_age": True,
    "age_weight": 0.15,  # effect per 1 SD of age (positive means older -> slightly higher event score)
    "use_hit_type": True,
    "hit_type_weight": 0.25,  # scales the mapping value below
    # hit type mapping; adjust names to match your data. Unseen types default to 0.0
    "hit_type_map": {
        # examples (tune for your dataset naming):
        "home_run": 1.0,
        "triple":   0.7,
        "double":   0.4,
        "single":   0.2,
        "walk":     0.1,
        "hit_by_pitch": 0.1,
        "ground_out": -0.2,
        "fly_out":    -0.2,
        "strikeout":  -0.4,
        "out":        -0.3,
        # fallback for anything else is 0.0
    },
    "min_events_per_player": 6,  # drop players with too few events to be rated robustly
}


# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    events = load_events(INPUT_CSV)
    # pick season
    df_s, season_used = choose_season(events, SEASON)

    players, events_by_player = build_event_scores(df_s, CONFIG)
    if len(players) < 2:
        raise SystemExit("Not enough eligible players after filtering (min_events_per_player).")

    ratings_all = run_elo_sim(
        players=players,
        events_by_player=events_by_player,
        n_sims=N_SIMULATIONS,
        matches_per_sim=MATCHES_PER_SIM,
        k=K_FACTOR,
        init_rating=INITIAL_RATING,
        seed=48
    )

    results = summarize_elo(players, ratings_all, INITIAL_RATING)

    # save
    out_csv = os.path.join(OUT_DIR, f"player_elo_{season_used}.csv")
    out_pdf = os.path.join(OUT_DIR, f"player_elo_{season_used}.pdf")
    results.to_csv(out_csv, index=False)
    plot_elo(results, season_used, INITIAL_RATING, out_pdf)

    # also save raw simulations if you want to inspect CIs precisely
    raw_np = pd.DataFrame(ratings_all, columns=players)
    raw_np.to_csv(os.path.join(OUT_DIR, f"player_elo_draws_{season_used}.csv"), index=False)

    print(f"Saved Elo table -> {out_csv}")
    print(f"Saved Elo plot  -> {out_pdf}")
    print(f"Players rated: {len(players)} | Simulations: {N_SIMULATIONS} | Matches/sim: {MATCHES_PER_SIM}")
