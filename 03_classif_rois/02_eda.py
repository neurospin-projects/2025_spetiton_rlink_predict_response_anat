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

INPUT_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'

################################################################################
# %% Load Data
# ============

data = pd.read_csv(INPUT_DATA)

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

# Reorder features by grouping by ROI and plotting correlation again with reordered features
from ml_utils import group_by_roi

roi_groups = group_by_roi(feature_columns)
ordered_features = [feat for features in roi_groups.values() for feat in features]
X_df = X_df[ordered_features]
corr_mat, pub_corr                     = plot_correlation(X_df[ordered_features],figscale=5.0, filename="reports/eda_correlation.png")
cluster_df, pub_dend                   = plot_feature_dendrogram(X_df, filename="reports/eda_dendrogram.png")
pub_corr_reordered = pub_dend


corr_mat_reordered, _           = plot_correlation(X_df[ordered_features], figscale=5.0, filename="reports/eda_correlation_reordered.png")
corr_mat_reordered.columns

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
scree_df_ha, _, _, _   = plot_pca_components(X_df_ha, filename="reports/eda_pca_components_hipp_amyg.png")



################################################################################
# %% PCA — global effect of age, sex and site on features
# ========================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils.stats_pairwise import pairwise_stats

# ── Fit PCA on standardised features ──────────────────────────────────────────
X_imp = SimpleImputer(strategy="median").fit_transform(X)
X_sc  = StandardScaler().fit_transform(X_imp)
pca   = PCA(n_components=2)
PCs   = pca.fit_transform(X_sc)

pca_df = pd.DataFrame(PCs, columns=["PC1", "PC2"])
pca_df["participant_id"] = data["participant_id"].values
pca_df["age"]  = data["age"].values
pca_df["sex"]  = data["sex"].values
pca_df["site"] = data["site"].values
ev = pca.explained_variance_ratio_
print(f"PC1: {ev[0]*100:.1f}%   PC2: {ev[1]*100:.1f}%   total: {(ev[0]+ev[1])*100:.1f}%")

# ── Outlier detection via Mahalanobis distance on PC1–PC2 ─────────────────────
from scipy.stats import chi2 as _chi2

pc_scores  = pca_df[["PC1", "PC2"]].values
mu         = pc_scores.mean(axis=0)
cov        = np.cov(pc_scores, rowvar=False)
cov_inv    = np.linalg.inv(cov)
diff       = pc_scores - mu
mahal_sq   = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)   # (n,)

# Chi-squared threshold: df=2 (number of PCs), p=0.001
alpha_out  = 0.001
threshold  = _chi2.ppf(1 - alpha_out, df=2)
is_outlier = mahal_sq > threshold

pca_df["mahal_sq"]   = mahal_sq
pca_df["is_outlier"] = is_outlier

outliers_df = pca_df[is_outlier][["participant_id", "PC1", "PC2", "mahal_sq"]].sort_values("mahal_sq", ascending=False)
print(f"\nOutliers detected (Mahalanobis χ²>{threshold:.1f}, p<{alpha_out}): n={is_outlier.sum()}")
print(outliers_df.to_string(index=False))

# Scatter with outliers highlighted
fig_out, ax_out = plt.subplots(figsize=(6, 5))
ax_out.scatter(pca_df.loc[~is_outlier, "PC1"], pca_df.loc[~is_outlier, "PC2"],
               color="#4878d0", alpha=0.7, s=40, label="normal")
ax_out.scatter(pca_df.loc[is_outlier,  "PC1"], pca_df.loc[is_outlier,  "PC2"],
               color="#d62728", alpha=0.9, s=80, marker="x", linewidths=2, label="outlier")
for _, row in outliers_df.iterrows():
    ax_out.annotate(str(row["participant_id"]),
                    (row["PC1"], row["PC2"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
ax_out.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
ax_out.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax_out.set_title(f"PCA outlier detection (Mahalanobis, α={alpha_out})")
ax_out.legend()
plt.tight_layout()
plt.savefig("reports/eda_pca_outliers.png", dpi=150, bbox_inches="tight")
plt.savefig("reports/eda_pca_outliers.pdf", bbox_inches="tight")
plt.show()

outliers_df.to_csv("reports/eda_pca_outliers.csv", index=False)
print("Saved: reports/eda_pca_outliers.png / .pdf / .csv")

# ── Refit PCA on outlier-free dataset ─────────────────────────────────────────
mask_clean   = ~is_outlier
X_clean      = X_imp[mask_clean]
data_clean   = data[mask_clean].reset_index(drop=True)
y_clean      = y[mask_clean].reset_index(drop=True)

X_sc_clean   = StandardScaler().fit_transform(X_clean)
pca_clean    = PCA(n_components=2)
PCs_clean    = pca_clean.fit_transform(X_sc_clean)
ev_clean     = pca_clean.explained_variance_ratio_

pca_df = pd.DataFrame(PCs_clean, columns=["PC1", "PC2"])
pca_df["participant_id"] = data_clean["participant_id"].values
pca_df["age"]      = data_clean["age"].values
pca_df["sex"]      = data_clean["sex"].values
pca_df["site"]     = data_clean["site"].values
pca_df["response"] = y_clean.values
ev = ev_clean
print(f"\nOutlier-free PCA — PC1: {ev[0]*100:.1f}%  PC2: {ev[1]*100:.1f}%  "
      f"n={mask_clean.sum()} participants")

# ── 2×2 scatter on outlier-free PCA: age / sex / site / response ──────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

def _pc_labels(ax):
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")

# age (continuous colormap)
sc = axes[0].scatter(pca_df["PC1"], pca_df["PC2"],
                     c=pca_df["age"], cmap="viridis", alpha=0.8, s=40)
plt.colorbar(sc, ax=axes[0], label="Age")
axes[0].set_title("Coloured by age")
_pc_labels(axes[0])

# sex
sex_labels  = pca_df["sex"].astype(str)
palette_sex = dict(zip(sorted(sex_labels.unique()), ["#4878d0", "#ee854a"]))
for val, grp in pca_df.groupby(sex_labels):
    axes[1].scatter(grp["PC1"], grp["PC2"],
                    c=palette_sex[str(val)], label=f"sex={val}", alpha=0.8, s=40)
axes[1].set_title("Coloured by sex")
axes[1].legend()
_pc_labels(axes[1])

# site
site_labels  = pca_df["site"].astype(str)
site_vals    = sorted(site_labels.unique())
palette_site = dict(zip(site_vals, sns.color_palette("tab10", len(site_vals))))
for val, grp in pca_df.groupby(site_labels):
    axes[2].scatter(grp["PC1"], grp["PC2"],
                    c=[palette_site[str(val)]], label=f"site={val}", alpha=0.8, s=40)
axes[2].set_title("Coloured by site")
axes[2].legend()
_pc_labels(axes[2])

# response (target y)
resp_labels  = pca_df["response"].astype(str)
palette_resp = dict(zip(sorted(resp_labels.unique()), ["#d62728", "#2ca02c"]))
resp_names   = {str(k): v for k, v in config.get("target_remap_inv",
                {0: "non-responder", 1: "responder"}).items()}
for val, grp in pca_df.groupby(resp_labels):
    label = resp_names.get(str(val), f"y={val}")
    axes[3].scatter(grp["PC1"], grp["PC2"],
                    c=palette_resp[str(val)], label=label, alpha=0.8, s=40)
axes[3].set_title("Coloured by response")
axes[3].legend()
_pc_labels(axes[3])

plt.suptitle(f"PC1–PC2 scatter plots",
             fontsize=14, fontweight="bold", y=1.01)
# plt.suptitle(f"PC1–PC2 scatter plots (outliers removed, n={mask_clean.sum()})",
#             fontsize=14, fontweight="bold", y=1.01)plt.tight_layout()
plt.savefig("reports/eda_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.savefig("reports/eda_pca_scatter.pdf", bbox_inches="tight")
plt.show()

# ── Association of PC1, PC2 with age, sex, site and features ──────────────────
assoc_df = pca_df[["PC1", "PC2", "age", "sex", "site"]].copy()

pca_stats_df = pairwise_stats(
    assoc_df,
    vars1=["PC1", "PC2"],
    vars2=["age", "sex", "site"],
    cattest="prop_ztest",
)
print("\nPC1–PC2 associations with covariates:")
print(pca_stats_df[["v1", "v2", "test", "stat", "dof", "pval", "descriptive"]].to_string(index=False))

pca_stats_df.to_excel("reports/eda_pca_stats.xlsx", index=False)
pca_stats_df.to_csv("reports/eda_pca_stats.csv", index=False)
print("Saved: reports/eda_pca_stats.xlsx / .csv")

# %%
