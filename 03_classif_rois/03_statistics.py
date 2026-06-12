"""
Univariate statistics
"""


import pandas as pd
import numpy as np
import os

from utils.stats_pairwise import pairwise_stats
from config import config
import seaborn as sns

#INPUT_CAT12_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'
INPUT_CAT12_DATA = './data/processed/roi-cat12vbm/df_ROI-notscaled_age_sex_site_M00_v4labels.csv'
INPUT_FS_DATA = './data/processed/fs_aseg_volumes.tsv'
PARTICIPANTS_DATA= './data/participants.tsv'
RESPONSE_DATA= './data/dataset-outcome_version-4.tsv'
# VBM
FEATURES_HIPAMY_VBM = ['Left_Hippocampus_GM_Vol', 'Right_Hippocampus_GM_Vol',
                'Left_Amygdala_GM_Vol', 'Right_Amygdala_GM_Vol']
# FreeSurfer
FEATURES_HIPAMY_FS = ['Left_Hippocampus', 'Right_Hippocampus',
                'Left_Amygdala', 'Right_Amygdala']


OUTPUT = "reports/classification_fs_reports.xlsx"

def load_vbm_data(features = []):
    data_cat12 = pd.read_csv(INPUT_CAT12_DATA, index_col="participant_id")
    return data_cat12


def load_response():
    response = pd.read_csv(RESPONSE_DATA, index_col="participant_id", sep='\t')
    label = 'Response.Status.at.end.of.follow.up'
    response = response[[label]]
    response = response.rename(columns={label:"response"})
    response = response[response['response'].isin(["NR","PaR","GR"])]
    return response


def load_fs_data(features = []):
    data_fs = pd.read_csv(INPUT_FS_DATA, index_col="participant_id", sep="\t")
    data_fs = data_fs[data_fs.session == "M00"]
    data_fs.drop(columns=["session"], inplace=True)
    data_fs = data_fs
    
    # age, sex from participants
    participants = pd.read_csv(PARTICIPANTS_DATA, index_col="participant_id", sep='\t')[["age", "sex", "ses-M00_center"]]
    participants = participants.rename(columns={"ses-M00_center":'site'})
    
    data_fs = pd.merge(data_fs, participants, left_index=True, right_index=True, how="left",)
    data_fs['site'] = data_fs['site'].astype(int).apply(lambda x: f"site-{x:02d}")
    
    # Response
    response = load_response()
    data_fs = pd.merge(data_fs, response, left_index=True, right_index=True, how="left",)
    assert data_fs.shape[0] == 135
    data_fs = data_fs[data_fs['response'].isin(["NR","PaR","GR"])]
    
    return data_fs


    # column of population dataframe that defines response to Li label
    
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
    return stats, lmfit

################################################################################
# %% Load Data
# ============

def load_data():
    from utils.pandas_utils import merge_on_index

    participants = pd.read_csv(PARTICIPANTS_DATA, index_col="participant_id", sep='\t')[["age", "sex"]]

    # VBM
    data_vbm = load_vbm_data()
    data_vbm["response"] = data_vbm["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})

    assert data_vbm.shape[0] == 117

    print(data_vbm.groupby('sex')['tiv'].mean())

    # Check participant_id, age and sex from data_vbm exactly match participants
    assert merge_on_index(data_vbm, participants, on= ["age", "sex"]).shape[0] == data_vbm.shape[0]

    data_vbm.columns = data_vbm.columns.str.replace(" ", "_")

    ## FS
    data_fs = load_fs_data()
    data_fs["response"] = data_fs["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})

    data_fs.columns = data_fs.columns.str.replace("-", "_")
    assert data_fs.shape[0] == 116

    global_features_fs = \
    ['BrainSeg', 'BrainSegNotVent', 'CerebralWhiteMatter', 'Cortex', 
    'EstimatedTotalIntraCranialVol', 'Mask', 'SubCortGray', 'SupraTentorial',
    'SupraTentorialNotVent', 'SurfaceHoles', 'TotalGray', 'VentricleChoroidVol', 
    'lhCerebralWhiteMatter', 'lhCortex', 'lhSurfaceHoles', 'rhCerebralWhiteMatter', 
    'rhCortex', 'rhSurfaceHoles']
    
    # Merge 
    data_vbm_fs = merge_on_index(data_vbm[merge_cols + FEATURES_HIPAMY_VBM],
                                data_fs[merge_cols + FEATURES_HIPAMY_FS], on=merge_cols)
    assert data_vbm_fs.shape[0] == 116
    import seaborn as sns

    return data_vbm, data_fs, data_vbm_fs
# data_fs = pd.merge(data_fs, data_vbm[["response", "age", "sex", "site"]],
#                    left_index=True, right_index=True, how='left')


# %% Pairwise Univariate statistics
# =================================

def stats_pairwise():
    stats_vbm_pairwise = pairwise_stats(data_vbm, vars1=['response'],
                        vars2=features_vbm + ['age', 'sex', 'site'],
                        cattest="prop_ztest")
    stats_vbm_pairwise.sort_values('pval', inplace=True)
    return stats_vbm_pairwise

# %% Linear regression univariate statistics
# ==========================================

s= "{feat} ~ response * sex + age + site + tiv"

def stats_lm(data_vbm, data_fs, formula, return_stats):
    
    # VBM
    return_stats = ["response", "age", "sex[T.M]"]
    stats_list = []
    for feat in FEATURES_HIPAMY_VBM:
        stats, lmfit = lm(
            formula=formula % feat,
            data=data_vbm,
            return_stats=return_stats
        )
        stats["var"] = feat
        stats_list.append(stats)

    stats_vbm = pd.concat(stats_list)
    stats_vbm = stats_vbm.reset_index(names="ivar")
    stats_vbm["Modality"] = "VBM"

    print(stats_vbm)

    # FS
    stats_list = []
    for feat in FEATURES_HIPAMY_FS:
        stats, lmfit = lm(
            formula=formula % feat,
            data=data_fs,
            return_stats=return_stats
        )
        stats["var"] = feat
        stats_list.append(stats)

    stats_fs = pd.concat(stats_list)
    stats_fs = stats_fs.reset_index(names="ivar")
    stats_fs["Modality"] = "FS"

    print(stats_fs)
    return stats_vbm, stats_fs


# %% ===========================================================================
if __name__ == '__main__':


    merge_cols = ["age", "sex", "response"]

        
    data_vbm, data_fs, data_vbm_fs = load_data()
    data_fs = data_fs.rename(columns={"EstimatedTotalIntraCranialVol":"tiv"})
    sns.violinplot(data=data_vbm, x='sex', y='tiv')
    sns.violinplot(data=data_fs, x='sex', y='tiv')
    sns.pairplot(data_vbm_fs, hue="response")

    features_vbm = [c for c in data_vbm.columns if c not in ["age", "sex", "response", "site"]]
    features_fs = [c for c in data_fs.columns if c not in ["age", "sex", "response", "site"]]

    
    stats_vbm_pairwise = stats_pairwise()
    stats_unscaled_vbm, stats_unscaled_fs = stats_lm(data_vbm, data_fs,
                formula="%s ~ response + sex + age + site + tiv",
                return_stats=["response", "age", "sex[T.M]"])
        
    stats_unscaled_interaction_vbm, stats_unscaled_interaction_fs = stats_lm(data_vbm, data_fs,
                formula="%s ~ response * sex + age + site + tiv",
                return_stats=["response", "age", "sex[T.M]", "response:sex[T.M]"])

    # Same analysis on scaled data
    target_tiv = 1500
    scaling_factor = target_tiv / data_vbm["tiv"]
    data_vbm_scaled = data_vbm.copy()
    data_vbm_scaled[features_vbm] = data_vbm[features_vbm].mul(scaling_factor, axis=0)
    assert np.allclose(data_vbm_scaled.tiv, target_tiv)

    target_tiv = 1500000
    scaling_factor = target_tiv / data_fs["tiv"]
    data_fs_scaled = data_fs.copy()
    data_fs_scaled[features_fs] = data_fs[features_fs].mul(scaling_factor, axis=0)
    assert np.allclose(data_fs_scaled.tiv, target_tiv)

    stats_scaled_vbm, stats_scaled_fs = stats_lm(data_vbm_scaled, data_fs_scaled,
        formula="%s ~ response + sex + age + site",
                return_stats=["response", "age", "sex[T.M]"])

    stats_scaled_interaction_vbm, stats_scaled_interaction_fs = stats_lm(data_vbm_scaled, data_fs_scaled,
        formula="%s ~ response * sex + age + site",
        return_stats=["response", "age", "sex[T.M]", "response:sex[T.M]"])

    with pd.option_context('display.max_rows', None):
        print(stats_scaled_interaction_vbm)
        
    with pd.option_context('display.max_rows', None):        
        print(stats_scaled_interaction_fs)
    
    # %% Save results
    excel_path = 'reports/statistics_univariate_interactions.xlsx'
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        stats_vbm_pairwise.to_excel(writer, sheet_name="stats_vbm_pairwise", index=False)
        stats_vbm.to_excel(writer, sheet_name="stats_vbm_lm", index=False)
        stats_fs.to_excel(writer, sheet_name="stats_fs_lm", index=False)
    print(f"Saved {excel_path}")

# %%

"""
ivar      coef         t             p                       var Modality
0   response -0.145760 -2.736881  7.365720e-03   Left_Hippocampus_GM_Vol      VBM
1        age -0.008384 -4.068872  9.566621e-05   Left_Hippocampus_GM_Vol      VBM
2   sex[T.M]  0.319919  6.052199  2.625430e-08   Left_Hippocampus_GM_Vol      VBM
3   response -0.163574 -3.166885  2.054375e-03  Right_Hippocampus_GM_Vol      VBM
4        age -0.008569 -4.287649  4.232130e-05  Right_Hippocampus_GM_Vol      VBM
5   sex[T.M]  0.299693  5.845922  6.622948e-08  Right_Hippocampus_GM_Vol      VBM
6   response -0.052887 -3.084419  2.650220e-03      Left_Amygdala_GM_Vol      VBM
7        age -0.002413 -3.636929  4.421435e-04      Left_Amygdala_GM_Vol      VBM
8   sex[T.M]  0.117362  6.896142  5.261136e-10      Left_Amygdala_GM_Vol      VBM
9   response -0.051079 -2.972360  3.718580e-03     Right_Amygdala_GM_Vol      VBM
10       age -0.002403 -3.613731  4.785267e-04     Right_Amygdala_GM_Vol      VBM
11  sex[T.M]  0.121608  7.129784  1.730536e-10     Right_Amygdala_GM_Vol      VBM
        ivar        coef         t         p                var Modality
0   response -147.217191 -2.121069  0.036492   Left_Hippocampus       FS
1        age  -13.355417 -4.969214  0.000003   Left_Hippocampus       FS
2   sex[T.M]  196.104456  2.279961  0.024823   Left_Hippocampus       FS
3   response -171.940496 -2.637213  0.009751  Right_Hippocampus       FS
4        age   -9.770353 -3.870003  0.000198  Right_Hippocampus       FS
5   sex[T.M]  104.732417  1.296259  0.197994  Right_Hippocampus       FS
6   response  -64.213369 -1.769974  0.079906      Left_Amygdala       FS
7        age   -4.684346 -3.334451  0.001216      Left_Amygdala       FS
8   sex[T.M]  185.069128  4.116415  0.000081      Left_Amygdala       FS
9   response  -89.795343 -2.679458  0.008678     Right_Amygdala       FS
10       age   -5.385410 -4.149980  0.000072     Right_Amygdala       FS
11  sex[T.M]  152.718992  3.677308  0.000389     Right_Amygdala       FS

With interaction

                 ivar      coef         t         p                       var Modality
0            response -0.138044 -2.291459  0.024123   Left_Hippocampus_GM_Vol      VBM
1                 age -0.008092 -4.960184  0.000003   Left_Hippocampus_GM_Vol      VBM
2            sex[T.M]  0.044354  0.712324  0.477992   Left_Hippocampus_GM_Vol      VBM
3   response:sex[T.M] -0.022344 -0.261897  0.793962   Left_Hippocampus_GM_Vol      VBM
4            response -0.126278 -2.164366  0.032917  Right_Hippocampus_GM_Vol      VBM
5                 age -0.008180 -5.177109  0.000001  Right_Hippocampus_GM_Vol      VBM
6            sex[T.M]  0.053823  0.892522  0.374345  Right_Hippocampus_GM_Vol      VBM
7   response:sex[T.M] -0.080490 -0.974136  0.332437  Right_Hippocampus_GM_Vol      VBM
8            response -0.040817 -2.047306  0.043360      Left_Amygdala_GM_Vol      VBM
9                 age -0.002289 -4.239370  0.000052      Left_Amygdala_GM_Vol      VBM
10           sex[T.M]  0.039792  1.930988  0.056435      Left_Amygdala_GM_Vol      VBM
11  response:sex[T.M] -0.025998 -0.920776  0.359475      Left_Amygdala_GM_Vol      VBM
12           response -0.034962 -1.751299  0.083088     Right_Amygdala_GM_Vol      VBM
13                age -0.002265 -4.188677  0.000062     Right_Amygdala_GM_Vol      VBM
14           sex[T.M]  0.046912  2.273509  0.025224     Right_Amygdala_GM_Vol      VBM
15  response:sex[T.M] -0.033985 -1.202057  0.232298     Right_Amygdala_GM_Vol      VBM

                 ivar      coef         t         p                       var Modality
0            response -0.138044 -2.291459  0.024123   Left_Hippocampus_GM_Vol      VBM
1                 age -0.008092 -4.960184  0.000003   Left_Hippocampus_GM_Vol      VBM
2            sex[T.M]  0.044354  0.712324  0.477992   Left_Hippocampus_GM_Vol      VBM
3   response:sex[T.M] -0.022344 -0.261897  0.793962   Left_Hippocampus_GM_Vol      VBM
4            response -0.126278 -2.164366  0.032917  Right_Hippocampus_GM_Vol      VBM
5                 age -0.008180 -5.177109  0.000001  Right_Hippocampus_GM_Vol      VBM
6            sex[T.M]  0.053823  0.892522  0.374345  Right_Hippocampus_GM_Vol      VBM
7   response:sex[T.M] -0.080490 -0.974136  0.332437  Right_Hippocampus_GM_Vol      VBM
8            response -0.040817 -2.047306  0.043360      Left_Amygdala_GM_Vol      VBM
9                 age -0.002289 -4.239370  0.000052      Left_Amygdala_GM_Vol      VBM
10           sex[T.M]  0.039792  1.930988  0.056435      Left_Amygdala_GM_Vol      VBM
11  response:sex[T.M] -0.025998 -0.920776  0.359475      Left_Amygdala_GM_Vol      VBM
12           response -0.034962 -1.751299  0.083088     Right_Amygdala_GM_Vol      VBM
13                age -0.002265 -4.188677  0.000062     Right_Amygdala_GM_Vol      VBM
14           sex[T.M]  0.046912  2.273509  0.025224     Right_Amygdala_GM_Vol      VBM
15  response:sex[T.M] -0.033985 -1.202057  0.232298     Right_Amygdala_GM_Vol      VBM
"""
