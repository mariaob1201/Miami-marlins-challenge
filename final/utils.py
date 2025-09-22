import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


# Load and prepare data
def load_and_prep_data(filepath):
    """Load and preprocess the exit velocity data"""
    df = pd.read_csv(filepath)

    # Create level hierarchy mapping - handle league class
    level_map = {
        'MLB': 0,
        'AAA': 1,
        'AA': 2
    }
    df['level_numeric'] = df['level_abbr'].map(level_map)
    df['level_numeric'] = df['level_numeric'].astype(int)

    # Encode categorical variables and store mappings
    df['batter_idx'], batter_mapping = pd.factorize(df['batter_id'])
    df['pitcher_idx'], pitcher_mapping = pd.factorize(df['pitcher_id'])
    df['hit_type_idx'], hit_type_mapping = pd.factorize(df['hit_type'])
    df['pitch_group_idx'], pitch_group_mapping = pd.factorize(df['pitch_group'])

    # Center age around mean
    df['age_centered'] = df['age'] - df['age'].mean()

    # Create handedness matchup (0=same, 1=opposite)
    df['handedness_matchup'] = (df['batter_hand'] != df['pitcher_hand']).astype(int)

    # Store all mappings in a dictionary for easy access
    mappings = {
        'batter_mapping': batter_mapping,
        'pitcher_mapping': pitcher_mapping,
        'hit_type_mapping': hit_type_mapping,
        'pitch_group_mapping': pitch_group_mapping
    }

    return df, mappings