# Miami Marlins — Exit Velocity Projection

Estimate each hitter's **underlying average exit velocity against MLB-level
competition** and project it into the 2024 season, for all 647 batters in
`exit_velo_validate_data.csv`.

> **TL;DR** — The recommended model is a **hierarchical Bayesian** projection
> ([`3_Modeling/hierarchical/`](3_Modeling/hierarchical/)). On a strict 2023 holdout it
> achieves **RMSE 1.96 mph**, ~19% better than the best naive baseline and far ahead
> of a per-batter ARIMA approach (3.18). It is the primary submission; the ARIMA
> model is retained as a documented baseline.

---

## Why a hierarchical model

Two facts about the data make partial pooling the right tool:

1. **Sparsity.** Of 3,715 batters, 1,642 have a single season and per-season sample
   sizes range from 1 to ~700 batted balls. A noisy 12-BBE estimate must be trusted
   *less* than a 600-BBE one — exactly what shrinkage provides.
2. **Cross-level projection.** ~100 of the 647 target batters have only minor-league
   history, so projecting their MLB ability requires an **estimated** level
   adjustment, not a hard-coded constant.

The model is fit on `(batter, season, level)` cells with a **measurement-error
likelihood** — each cell mean enters as `Normal(μ, se)` with `se = sd/√n`, so
sample-size weighting is built in. It estimates:

```
μ_cell = μ_global
       + θ_batter[b]            # latent ability  (partially pooled / shrunk)
       + η_batter_season        # season "form" intercept (identifies level effect)
       + α_level[level]         # level equivalency, MLB = reference
       + age curve (centered)
```

A batter's **2024 MLB ability** is `μ_global + θ_batter + α_MLB(0) + age curve(2024 age)`,
with a 95% credible interval from the posterior. Cold-start batters (no history) pool
to the population mean with appropriately wide intervals.

### Key findings

* **Level equivalencies (estimated, not assumed):** for the *same* hitter in the *same*
  season, exit velocity is **higher in the minors than MLB** — AAA `+1.26` mph
  `[1.17, 1.36]`, AA `+1.44` `[1.31, 1.57]`. Projecting a minor-league hitter to MLB
  therefore *subtracts* ~1.3 mph. Note this contradicts the hard-coded
  `LEVEL_EQUIV = {MLB:0, AAA:-2.5, AA:-4.5}` in the ARIMA baseline, which has the
  **opposite sign**. (The clean within-batter-season contrast is the cause: the raw
  cross-season contrast flips sign due to selection/development confounding.)
* **Uncertainty is honest:** mean 95% interval width is 3.3 mph for batters with
  history vs 15.0 mph for cold-start batters.
* Convergence: R-hat ≤ 1.006, 0 divergences (~90 s on 4 chains).

### 2023 holdout backtest

Train on `season ≤ 2022`; predict observed 2023 MLB mean for the 490 batters with
≥50 MLB batted balls in 2023 and prior history. Same batter set for every method.

| method        | RMSE | MAE  | bias  | corr |
|---------------|------|------|-------|------|
| **hierarchical** | **1.96** | **1.48** | −0.11 | **0.73** |
| career_mlb (naive) | 2.41 | 1.69 | +0.08 | 0.64 |
| last_mlb (naive)   | 2.58 | 1.82 | −0.19 | 0.62 |
| global_mean        | 2.85 | 2.20 | +0.28 | 0.00 |
| arima_ts           | 3.18 | 2.06 | +0.42 | 0.53 |

Per-batter ARIMA is the **worst** method — on 3–4 noisy annual points it overfits and
extrapolates spurious trends. Partial pooling wins decisively.

---

## Repository structure

```
1_Data/                              # exit_velo_project_data.parquet, exit_velo_validate_data.csv
2_Data_Exploration/                  # data_exploration.ipynb, data_to_parquet.py
3_Modeling/
  hierarchical/                      # ★ PRIMARY MODEL
    hierarchical_model.py            #   fit + project 2024 (PyMC)
    backtest_2023.py                 #   strict 2023 holdout vs baselines
    outputs/                         #   predictions, parameter_summary, reports
  time_series/                       # ARIMA baseline (arima.py, rev.ipynb)
  other_approaches/                  # exploratory: level equivalencies, deprecated drafts
Miami_marlins.pdf                    # written report
```

## How to run

```bash
pip install -r requirements.txt

# Primary model: fit + 2024 projections for all 647 validation batters
python 3_Modeling/hierarchical/hierarchical_model.py

# Reproduce the 2023 holdout backtest table
python 3_Modeling/hierarchical/backtest_2023.py
```

Outputs land in `3_Modeling/hierarchical/outputs/`:

| file | contents |
|------|----------|
| `per_batter_2024_predictions.csv` | 2024 MLB exit-velo projection + 95% interval, all 647 batters |
| `parameter_summary.csv` | posterior summaries (level effects, age curve, variances) |
| `summary_report.txt` | headline results + convergence diagnostics |
| `backtest_2023_metrics.csv` | the holdout comparison table above |

> The full posterior trace (`idata.nc`, ~600 MB) is git-ignored. Re-run with
> `SAVE_TRACE=1` to keep it locally for deeper diagnostics.

## Limitations & future work

* **Quality of competition.** Pitcher identity/handedness are not yet in the primary
  model. A batted-ball-level extension with pitcher random effects (drafted in
  `other_approaches/final_deprecated/`) is the natural next step.
* **Within-season trajectory.** The model treats ability as season-constant plus a
  population age curve; individual aging/development paths are not modeled.
* **Target noise.** The 2023 backtest target is itself a finite-sample mean; the ≥50-BBE
  filter mitigates but does not remove this.
