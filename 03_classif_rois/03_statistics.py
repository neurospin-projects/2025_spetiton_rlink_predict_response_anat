"""
Univariate statistics
"""


import pandas as pd
import numpy as np
import os

from utils.stats_pairwise import pairwise_stats
from config import config
import seaborn as sns

INPUT_CAT12_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'
INPUT_FS_DATA = './data/processed/fs_aseg_volumes.tsv'

OUTPUT = "reports/classification_fs_reports.xlsx"

def load_vbm_data(features = []):
    data_cat12 = pd.read_csv(INPUT_CAT12_DATA, index_col="participant_id")
    return data_cat12


def load_fs_data(features = []):
    data_fs = pd.read_csv(INPUT_FS_DATA, index_col="participant_id", sep="\t")
    data_fs = data_fs[data_fs.session == "M00"]
    data_fs.drop(columns=["session"], inplace=True)
    data_fs = data_fs
    return data_fs


def lm(formula, data, return_stats=[]):
    import statsmodels.formula.api as sm
    model = sm.ols(formula=formula, data=data)
    lmfit = model.fit()
    # or all at once as a clean DataFrame
    stats = pd.DataFrame({
        "coef":    lmfit.params[return_stats],
        "t":       lmfit.tvalues[return_stats],
        "p":       lmfit.pvalues[return_stats],
    })
    return stats

################################################################################
# %% Load Data
# ============

data_vbm = load_vbm_data()
data_vbm.columns = data_vbm.columns.str.replace(" ", "_")
data_vbm.rename(columns={"y": "response"}, inplace=True)
data_vbm["response"] = data_vbm["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})

# Select Input = dataframe - (target + drop + residualization)
features_vbm = [c for c in data_vbm.columns if c not in ["response"] + \
    config['drop'] + config['residualization']]

data_fs = load_fs_data()
data_fs.columns = data_fs.columns.str.replace("-", "_")

data_fs = pd.merge(data_fs, data_vbm[["response", "age", "sex", "site"]],
                   left_index=True, right_index=True, how='left')

global_features_fs = \
['BrainSeg', 'BrainSegNotVent', 'CerebralWhiteMatter', 'Cortex', 
 'EstimatedTotalIntraCranialVol', 'Mask', 'SubCortGray', 'SupraTentorial',
 'SupraTentorialNotVent', 'SurfaceHoles', 'TotalGray', 'VentricleChoroidVol', 
 'lhCerebralWhiteMatter', 'lhCortex', 'lhSurfaceHoles', 'rhCerebralWhiteMatter', 
 'rhCortex', 'rhSurfaceHoles']
 
sns.violinplot(data=data_fs, x='sex', y='EstimatedTotalIntraCranialVol')

# %% Pairwise Univariate statistics
# =================================

stats_pairwise = pairwise_stats(data_vbm, vars1=['response'],
                       vars2=features_vbm + ['age', 'sex', 'site'], cattest="prop_ztest")
stats_pairwise.sort_values('pval', inplace=True)

# %% Linear regression univariate statistics
# ==========================================

return_stats = ["response", "age", "sex"]

# VBM
features_vbm = ['Left_Hippocampus_GM_Vol', 'Right_Hippocampus_GM_Vol',
                'Left_Amygdala_GM_Vol', 'Right_Amygdala_GM_Vol']


formula = "Left_Hippocampus_GM_Vol ~ response + age + sex + site"


stats_list = []
for feat in features_vbm:
    stats = lm(
        formula=f"{feat} ~ response + age + sex + site",
        data=data_vbm,
        return_stats=["response", "age", "sex"]
    )
    stats["var"] = feat
    stats_list.append(stats)

stats_vbm = pd.concat(stats_list)
stats_vbm = stats_vbm.reset_index(names="ivar")
stats_vbm["Modality"] = "VBM"

print(stats_vbm)

# FreeSurfer
features_fs = ['Left_Hippocampus', 'Right_Hippocampus',
                'Left_Amygdala', 'Right_Amygdala']


stats_list = []
for feat in features_fs:
    stats = lm(
        formula=f"{feat} ~ response + age + sex + site + EstimatedTotalIntraCranialVol",
        data=data_fs,
        return_stats=["response", "age", "sex"]
    )
    stats["var"] = feat
    stats_list.append(stats)

stats_fs = pd.concat(stats_list)
stats_fs = stats_fs.reset_index(names="ivar")
stats_fs["Modality"] = "FS"

print(stats_fs)

# %% Save results
excel_path = 'reports/statistics_univariate.xlsx'
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    stats_pairwise.to_excel(writer, sheet_name="statistics_paiwise", index=False)
    stats_vbm.to_excel(writer, sheet_name="statistics_vbm_lm", index=False)
    stats_fs.to_excel(writer, sheet_name="statistics_fs_lm", index=False)
print(f"Saved {excel_path}")

# %%

"""
ivar      coef         t         p                       var Modality
0   response -0.145219 -3.194811  0.001883   Left_Hippocampus_GM_Vol      VBM
1        age -0.007889 -4.485544  0.000020   Left_Hippocampus_GM_Vol      VBM
2        sex  0.099682  2.209510  0.029468   Left_Hippocampus_GM_Vol      VBM
3   response -0.160183 -3.569390  0.000556  Right_Hippocampus_GM_Vol      VBM
4        age -0.007766 -4.472696  0.000021  Right_Hippocampus_GM_Vol      VBM
5        sex  0.146691  3.293361  0.001378  Right_Hippocampus_GM_Vol      VBM
6   response -0.051080 -3.666606  0.000399      Left_Amygdala_GM_Vol      VBM
7        age -0.002204 -4.088766  0.000089      Left_Amygdala_GM_Vol      VBM
8        sex  0.009433  0.682198  0.496723      Left_Amygdala_GM_Vol      VBM
9   response -0.048743 -3.595349  0.000509     Right_Amygdala_GM_Vol      VBM
10       age -0.002191 -4.176680  0.000064     Right_Amygdala_GM_Vol      VBM
11       sex  0.002591  0.192536  0.847721     Right_Amygdala_GM_Vol      VBM
        ivar        coef         t         p                var Modality
0   response -147.217191 -2.121069  0.036492   Left_Hippocampus       FS
1        age  -13.355417 -4.969214  0.000003   Left_Hippocampus       FS
2        sex -196.104456 -2.279961  0.024823   Left_Hippocampus       FS
3   response -171.940496 -2.637213  0.009751  Right_Hippocampus       FS
4        age   -9.770353 -3.870003  0.000198  Right_Hippocampus       FS
5        sex -104.732417 -1.296259  0.197994  Right_Hippocampus       FS
6   response  -64.213369 -1.769974  0.079906      Left_Amygdala       FS
7        age   -4.684346 -3.334451  0.001216      Left_Amygdala       FS
8        sex -185.069128 -4.116415  0.000081      Left_Amygdala       FS
9   response  -89.795343 -2.679458  0.008678     Right_Amygdala       FS
10       age   -5.385410 -4.149980  0.000072     Right_Amygdala       FS
11       sex -152.718992 -3.677308  0.000389     Right_Amygdala       FS
"""
