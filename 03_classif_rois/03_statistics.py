"""
Univariate statistics
"""


import pandas as pd
import numpy as np
import os

from utils.stats_pairwise import pairwise_stats
from ml_utils import get_residualizer, create_print_log
from config import config

################################################################################
# %% Load Data
# ============

data = pd.read_csv(config['input_data'])

# Select Input = dataframe - (target + drop + residualization)
feature_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]

X = data[feature_columns].values
y = data[config['target']].map(config['target_remap'])

# Multiply CSF columns by -1
csf_indices = np.array([i for i, col in enumerate(feature_columns) if 'CSF' in col])
X[:, (csf_indices)] *= -1



# %% Run Univariate statistics
# ============================

# Create DataFrame for stats
X_df = pd.DataFrame(X, columns=feature_columns)
X_df["response"] = y.values
X_df["age"] = data["age"]
X_df["sex"] = data["sex"]
X_df["site"] = data["site"]


stats = pairwise_stats(X_df, vars1=['response'],
                       vars2=feature_columns + ['age', 'sex', 'site'], cattest="prop_ztest")
stats.sort_values('pval', inplace=True)

excel_path = 'reports/statistics_univariate.xlsx'
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    stats.to_excel(writer, sheet_name="statistics_univariate", index=False)
print(f"Saved {excel_path}")
