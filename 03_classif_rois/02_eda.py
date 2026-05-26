"""
Exploratory Data Analysis (EDA) — Lithium Response
===============================================================
Runs the full EDA pipeline on the clinical feature matrix and saves
figures and a summary Excel workbook to reports/.

Steps
-----
1. Class balance — prints lithium-response group sizes.
2. Descriptive statistics — mean/SD/skewness for continuous variables;
   frequencies/proportions for binary/categorical variables.
3. Clustered correlation heatmap — hierarchical clustering of features
   based on absolute Pearson correlation; saved to eda_correlation_clustermap.png.
4. Pearson correlation matrix — plotted in the cluster order from step 3;
   saved to eda_correlation.png.
5. Variance Inflation Factors (VIF) — multicollinearity diagnosis;
   saved to eda_vif.png.
6. Feature dendrogram — Ward linkage on the VIF-standardised matrix;
   saved to eda_dendrogram.png.
7. PCA scree plot — explained variance vs. number of components with
   elbow detection; saved to eda_pca_components.png.
8. Feature–response associations — per-feature comparison between
   responders and non-responders; saved to eda_feature_response.png.

Outputs
-------
reports/eda_results.xlsx   — one sheet per analysis step, plus
                              ready-to-paste Methods / Results text.
reports/eda_*.png          — individual figures (one per step).

Inputs (from config.py)
-----------------------
data                    : pd.DataFrame — full patient dataset
config['clinical_vars'] : list[str]    — clinical feature names to analyse
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import utils
from utils.eda import (descriptive_stats, plot_correlation,
                       plot_feature_dendrogram, plot_pca_components)

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

################################################################################
# %% Exploratory Data Analysis
# ============================

# Create DataFrame for stats
X_df = pd.DataFrame(X, columns=feature_columns)
# X_df["response"] = y.values
# X_df["age"] = data["age"]
# X_df["sex"] = data["sex"]
# X_df["site"] = data["site"]

# ── Class balance ──────────────────────────────────────────────────────

counts = y.value_counts()
print("\n━━━  Lithium Response — Class Balance  ━━━")
for cls, cnt in counts.items():
    print(f"  Class {cls}: {cnt}  ({100*cnt/len(y):.1f} %)")
# ━━━  Lithium Response — Class Balance  ━━━
#   Class 0: 74  (63.2 %)
#   Class 1: 43  (36.8 %)

quant_df, cat_df, pub_desc             = descriptive_stats(data.drop(columns=['participant_id'], errors='ignore'), max_cat_unique=2)
corr_mat, pub_corr                     = plot_correlation(X_df, figscale=5.0, filename="reports/eda_correlation.png")
cluster_df, pub_dend                   = plot_feature_dendrogram(X_df, filename="reports/eda_dendrogram.png")
corr_mat_reordered, _           = plot_correlation(X_df[cluster_df["feature"]], figscale=5.0, filename="reports/eda_correlation_reordered.png")
pub_corr_reordered = pub_dend
scree_df, elbow_idx, thresh_results, pub_pca = plot_pca_components(X_df, filename="reports/eda_pca_components.png")


# ── Save all results to Excel ──────────────────────────────────────────────
pub_df = pd.DataFrame([
    {"function": "descriptive_stats",       **pub_desc},
    {"function": "plot_correlation",        **pub_corr},
    {"function": "plot_feature_dendrogram", **pub_dend},
    {"function": "plot_pca_components",     **pub_pca},
])

excel_path = "reports/eda_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    quant_df.to_excel(writer,           sheet_name="desc_quantitative")
    cat_df.to_excel(writer,             sheet_name="desc_categorical",      index=False)
    corr_mat.to_excel(writer,           sheet_name="corr_pearson")
    corr_mat_reordered.to_excel(writer, sheet_name="corr_pearson_reordered")
    cluster_df.to_excel(writer,         sheet_name="feature_clusters",      index=False)
    scree_df.to_excel(writer,           sheet_name="pca_scree",             index=False)
    pub_df.to_excel(writer,             sheet_name="publication_text",      index=False)
print(f"\n✔  Saved {excel_path}")


# ── EDA restricted to Hippocampus / Amygdala columns ──────────────────────
# [c for c in feature_columns if 'Hippocampus' in c or 'Amygdala' in c]
hipp_amyg_cols = ['Right Amygdala_GM_Vol',
 'Left Amygdala_GM_Vol',
 'Right Amygdala_CSF_Vol',
 'Left Amygdala_CSF_Vol',
 'Right Hippocampus_GM_Vol',
 'Left Hippocampus_GM_Vol',
 'Right Hippocampus_CSF_Vol',
 'Left Hippocampus_CSF_Vol']

X_df_ha = X_df[hipp_amyg_cols]

# import importlib
# import utils.eda
# importlib.reload(utils.eda)
# from utils.eda import plot_correlation, plot_feature_dendrogram, plot_pca_components

corr_mat_ha, _         = plot_correlation(X_df_ha, annot=True, linewidths=1, figscale=1.0, filename="reports/eda_correlation_hipp_amyg.png")
cluster_df_ha, _       = plot_feature_dendrogram(X_df_ha, filename="reports/eda_dendrogram_hipp_amyg.png")
corr_mat_reordered_ha, _ = plot_correlation(X_df_ha[cluster_df_ha["feature"]], annot=True, linewidths=1, figscale=1.0, filename="reports/eda_correlation_reordered_hipp_amyg.png")
scree_df_ha, _, _, _   = plot_pca_components(X_df_ha, filename="reports/eda_pca_components_hipp_amyg.png")


