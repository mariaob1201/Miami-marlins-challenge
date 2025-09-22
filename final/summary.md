# Hierarchical Bayesian Exit Velocity Talent Model
**Miami Marlins Challenge - Player Performance Analytics**  
**Author:** Maria Oros  
**Date:** September 21, 2025

## Executive Summary
This analysis develops a sophisticated hierarchical Bayesian framework to evaluate batter exit velocity talent, separating true underlying ability from noise and contextual factors. The model processes data from 3,715 batters across MLB, AAA, and AA levels.

### Key Findings
- **League Equivalencies:** AAA performance translates to approximately 2.5 mph lower exit velocity than MLB, while AA shows a 4.5 mph gap
- **Age Effects:** Peak performance occurs around age 27-28, with quadratic decline thereafter  
- **Shrinkage Benefits:** Players with limited data show 40-60% shrinkage toward population mean, improving projection accuracy
- **Uncertainty Quantification:** Model provides credible intervals reflecting confidence in each projection

## Setup and Data Loading

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from utils import *
from age_functions import *
from hierarchical_model import *

# Set plotting style
plt.style.use('default')  # Use default since seaborn-v0_8 might not be available
sns.set_palette("husl")

# Load the data
df = pd.read_csv('1_Data/exit_velo_project_data.csv')
```

## Data Exploration and Model Justification

```{code-cell} ipython3
# Analyze dataset structure
df = pd.read_csv('1_Data/exit_velo_project_data.csv')
level_stats, obs_per_player = analyze_dataset_structure(df)
```

```{code-cell} ipython3
# Visualize the distribution of exit velocities by league
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='level_abbr', y='exit_velo')
plt.title('Exit Velocity Distribution by League')
plt.ylabel('Exit Velocity (mph)')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='exit_velo', hue='level_abbr', alpha=0.7, bins=30)
plt.title('Exit Velocity Histograms by League')
plt.xlabel('Exit Velocity (mph)')

plt.tight_layout()
plt.savefig('fig2.html', bbox_inches='tight')
```

```{raw} html
display('fig2.html')
```



```{code-cell} ipython3
# Age effect analysis
df = pd.read_csv('1_Data/exit_velo_project_data.csv')
fit_all, peak_age_overall = analyze_age_effects(df)
```

```{raw} html
print(fit_all.summary())
print(peak_age_overall)
```

# ARIMAX Time Series Model for Exit Velocity Forecasting
The source code for the ARIMA modeling approach can be find here `3_Modeling/time_series/arima.py`

## Model Overview
We use an **ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)** model to forecast each batter's 2024 exit velocity performance. This approach combines historical time series patterns with relevant external factors that influence performance.

## Model Components

### Core ARIMAX Structure
The ARIMAX model has three main components:
- **AR (p)**: Uses past exit velocity values to predict future performance
- **I (d)**: Accounts for trends by differencing the data to make it stationary
- **MA (q)**: Models the error terms from previous predictions
- **X (exogenous)**: Incorporates external factors that affect performance

### Key Input Features

#### 1. League Competition Level
- **AA, AAA, MLB shares**: Percentage of plate appearances at each level per season
- Captures the quality of competition faced by each batter
- Automatically adjusts forecasts based on expected 2024 playing level

#### 2. Batting Approach Metrics
- **Launch angle patterns**: Sweet spot rate (15-25°), angle consistency, streakiness
- **Spray angle distribution**: Pull/opposite field tendencies, directional consistency
- **Contact quality indicators**: Distance from optimal launch angles

#### 3. Batted Ball Profile
- **Hit type distribution**: Ground ball, line drive, fly ball, popup rates
- Reflects each batter's characteristic swing mechanics and approach

## Automatic Model Selection
We use **Auto-ARIMA** to automatically determine the optimal model structure:
- Systematically tests different combinations of AR, I, and MA terms
- Selects the best-performing model using information criteria (AIC)
- Eliminates the need for manual parameter tuning
- Prevents overfitting by balancing model complexity with performance

## Forecasting Process

### 1. Historical Analysis
- Aggregates batter performance data by season (2019-2023)
- Builds comprehensive feature profiles for each player
- Identifies individual time series patterns and trends

### 2. Model Training
- Fits ARIMAX model using each batter's historical exit velocity
- Incorporates league level, batting approach, and contact quality as predictive factors
- Automatically selects optimal model parameters

### 3. 2024 Projection
- Projects forward one season using the fitted model
- Assumes batter will play at their most recent competition level
- Carries forward non-league features (batting approach remains consistent)
- Provides point estimates with 95% confidence intervals

## Model Advantages

### Personalized Forecasts
- Individual model for each batter captures unique patterns
- Accounts for player-specific development trajectories
- Adapts to different career stages and performance trends

### Context-Aware Predictions
- Adjusts for competition level (AA → AAA → MLB progression)
- Incorporates mechanical and approach factors beyond raw performance
- Recognizes that identical exit velocities may indicate different skill levels across leagues

### Robust Methodology
- Time series approach handles irregular seasonal patterns
- Multiple validation techniques ensure model reliability
- Fallback mechanisms prevent prediction failures

## Output Interpretation
Each forecast includes:
- **Point Estimate**: Most likely 2024 average exit velocity
- **Confidence Interval**: Range of plausible outcomes (95% confidence)
- **Method Used**: ARIMAX order selected (e.g., ARIMAX(1,1,0))
- **Model Validation**: Cross-validation metrics where available

## Limitations & Considerations
- Assumes recent patterns continue into 2024
- Limited by available historical data (minimum 3 seasons required)
- Does not account for major swing changes or injuries
- Performance may vary for players with limited MLB experience
- 
```{code-cell} ipython3
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df_out = pd.read_csv("3_Modeling/time_series/outputs/per_batter_2024_arimax.csv")
df = df_out[df_out['forecast']>=0]
# Filter to the “last-level” scenario (typical summary view)
last = df[df["target_level_2024"] == df["last_level"]].copy()

# Choose the series to plot (use df["forecast"] if you want all scenarios)
vals = last["forecast"].dropna().values

# Plot histogram
plt.figure(figsize=(7,4))
plt.hist(vals, bins=20)  # adjust bins if needed
plt.title("Histogram of 2024 Forecasted Exit Velocity")
plt.xlabel("Forecast EV (mph)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig_forec_hist.png", dpi=180, bbox_inches="tight")   # <-- PNG (works)

# Save and/or display
plt.savefig("hist_forecast_2024.png", dpi=180, bbox_inches="tight")

levels = ["AA","AAA","MLB"]
data = [df.loc[df["target_level_2024"]==L, "forecast"].dropna().values for L in levels]

plt.figure(figsize=(7,4))
plt.boxplot(data, labels=levels, showmeans=True)
plt.title("2024 Forecast Distribution by Target Level")
plt.ylabel("Forecast EV (mph)")
plt.tight_layout()
plt.savefig("fig_forec.png", dpi=180, bbox_inches="tight")   # <-- PNG (works)

```

```{raw} html
display('fig_forec_hist.html')
```


```{raw} html
display('fig_forec.html')
```



## Forecasting Modeling: Hierarchical Bayesian Model Implementation

