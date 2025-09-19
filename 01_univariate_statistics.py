import re, sys, os
import numpy as np, pandas as pd
from utils import get_rois
from sklearn.preprocessing import StandardScaler
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer
from plots import plot_glassbrain
# Statmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

# inputs
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
VBMLOOKUP_FILE = "/drf/local/spm12/tpm/labels_Neuromorphometrics.xml"
ROOT = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR = ROOT+"data/processed/"
M0_DATA = DATA_DIR+"df_ROI_age_sex_site_M00_v4labels.csv"
M3_MINUS_M0_DATA = DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site_v4labels.csv"
M0M3_DATA=DATA_DIR+"df_ROI_age_sex_site_M00_M03_v4labels.csv"

# zscores from normative models
ZSCORES_PATH = ROOT+"reports/normative_modeling/"
M3_MINUS_M0_ZSCORES = ZSCORES_PATH+"df_zscores_M3minusM0_with_age_sex_site_normative_BLR_BigHC.csv"
M3_MINUS_M0_ZSCORES_RES_ON_SITE = ZSCORES_PATH+"df_zscores_M3minusM0_with_age_sex_site_res_on_site_normative_BLR_BigHC_site_residualized.csv"
M0_ZSCORES_RES_ON_SITE= ZSCORES_PATH+"df_zscores_M0_with_age_sex_site_normative_BLR_BigHC_site_residualized.csv"
M0_ZSCORES = ZSCORES_PATH+"df_zscores_M0_with_age_sex_site_normative_BLR_BigHC.csv"

FEAT_IMPTCE_RES_DIR = ROOT+"reports/feature_importance_results/"
HYP_AMYG_GM_ROIS = ['Right Amygdala_GM_Vol', 'Left Amygdala_GM_Vol','Right Hippocampus_GM_Vol', 'Left Hippocampus_GM_Vol']
HYP_AMYG_CSF_ROIS = ['Right Amygdala_CSF_Vol', 'Left Amygdala_CSF_Vol','Right Hippocampus_CSF_Vol', 'Left Hippocampus_CSF_Vol']

# outputs
STATS_UNIV_RES_DIR = ROOT+"reports/stats_univ_results/"


def transform_df(df):
    # Replace +- and spaces are not alowed in statsmodel formula
    df.columns = df.columns.str.replace('-', '_MINUS_') 
    df.columns = df.columns.str.replace('+', '_PLUS_') 
    string_dict = {s.replace(" ", "_"): s for s in list(df.columns)}
    df.columns = df.columns.str.replace(' ', '_') 
    # Add an underscore if the string starts with a number
    before_adding_underscores=list(df.columns)
    with_added_underscores = ['_' + s if re.match(r'^\d', s) else s for s in before_adding_underscores]
    df.columns = with_added_underscores
    # dict that maps new df column names to the original df column names 
    string_dict = dict(zip(with_added_underscores, string_dict.values()))
    return df, string_dict

def site_r2(df, col):
    # keep only sites with >2 subjects
    df = df[df.groupby("site")["site"].transform("count") > 3]

    full_model = smf.ols(f"{col} ~ age + C(sex) + C(site) + C(response)", data=df).fit()
    anova_table = anova_lm(full_model, typ=2)
    ss_site = anova_table.loc["C(site)", "sum_sq"]
    ss_resid = anova_table.loc["Residual", "sum_sq"]
    r2_site = ss_site / (ss_site + ss_resid)
    return r2_site


def mean_roi_vals_hip_amyg():

    dfM3minusM0 = pd.read_csv(M3_MINUS_M0_DATA) # 91 subjects
    df = pd.read_csv(M0M3_DATA) # 91 subjects
    dfM0=pd.read_csv(M0_DATA) # 117 subjects
    dfM3=df[df["session"]=="M03"].copy()
    dfM3.reset_index(inplace=True)

    scaler1 = StandardScaler()
    df1_z = dfM3minusM0.copy()
    df1_z[HYP_AMYG_GM_ROIS] = scaler1.fit_transform(dfM3minusM0[HYP_AMYG_GM_ROIS])
    df1_z, string_dict = transform_df(df1_z)

    scaler2 = StandardScaler()
    df2_z = dfM0.copy()
    df2_z[HYP_AMYG_GM_ROIS] = scaler2.fit_transform(dfM0[HYP_AMYG_GM_ROIS])
    df2_z, _ = transform_df(df2_z)

    scaler3 = StandardScaler()
    df3_z = dfM3.copy()
    df3_z[HYP_AMYG_GM_ROIS] = scaler3.fit_transform(dfM3[HYP_AMYG_GM_ROIS])
    df3_z, _ = transform_df(df3_z)
    dict_opposite={v:k for k,v in string_dict.items()}
    hyp_amyg_new_col_names = [dict_opposite.get(x, x) for x in HYP_AMYG_GM_ROIS]

    rows = []
    for c in hyp_amyg_new_col_names:
        r2_1 = site_r2(df1_z, c)
        r2_2 = site_r2(df2_z, c)
        r2_3 = site_r2(df3_z, c)

        rows.append({"column": string_dict[c], "R2_site_df3minusM0": r2_1, "R2_site_dfM0": r2_2, "R2_site_dfM3": r2_3})

    results = pd.DataFrame(rows)
    print(results)

def test_M0():
    df = pd.read_csv(M0_DATA)
    # df = pd.read_csv(M0_ZSCORES)
    df["response"] = df["response"].replace({"GR": 1, "PaR": 0, "NR": 0})

    # ===== residualize on site
    rois = [r for r in list(df.columns) if r.endswith("_GM_Vol") or r.endswith("_CSF_Vol")]
    residualizer = Residualizer(data=df[["age","sex","site","response"]], formula_res="site", formula_full="site + sex + age + response")
    Zres = residualizer.get_design_mat(df[["age","sex","site","response"]])
    X= df[rois].values
    residualizer.fit(X, Zres)
    X = residualizer.transform(X, Zres)
    df[rois]=X
    # ===== end residualize on site

    # for r in HYP_AMYG_GM_ROIS:
    #     num_positive = (df[r] > 0).sum()
    #     print(r, "  ",num_positive)

    # center age
    df["age"] = df["age"] - df["age"].mean()
    print(df["site"].value_counts())

    df, string_dict = transform_df(df)
    rois = [r for r in list(df.columns) if r.endswith("_GM_Vol")]# or r.endswith("_CSF_Vol")]
    
    rows = []
    val_of_interest =  "C(response)[T.1]"
    # df contains rois, participant_id, age, sex, and site columns
    for col in rois:
        formula = f"{col} ~ age + C(sex) + C(response)" # + C(site)
        model = smf.ols(formula, data=df).fit()
        roi_name = string_dict[col]
        pval = model.pvalues[val_of_interest] 
        tstat = model.tvalues[val_of_interest] 
        # if model.pvalues['C(sex)[T.male]']<=0.05: print(col, " sex pval: ",model.pvalues['C(sex)[T.male]'])
        # if model.pvalues['age']<=0.05 : print(col ," age pval:",model.pvalues['age'])

        rows.append({"ROI": roi_name,
                        "pvalue": pval,
                        "tstatistic": tstat})

    results_df = pd.DataFrame(rows, columns=["ROI", "pvalue", "tstatistic"])
    # print(results_df)

    # print only significant rows
    significant = results_df[results_df["pvalue"] <= 0.05]
    significant_sorted = significant.sort_values(by="pvalue", ascending=True)

    # with pd.option_context('display.max_rows', None):
    print("\nSignificant (all rows, sorted by p-value):\n", significant_sorted)
    print("nb of significant roi:",len(significant_sorted))

    print("\n",results_df[results_df["ROI"].isin(HYP_AMYG_GM_ROIS)])
    print("\n",results_df[results_df["ROI"].isin(HYP_AMYG_CSF_ROIS)])

    return results_df

def test_M3minusM0(intercept=True):
    df = pd.read_csv(M3_MINUS_M0_DATA)
    # df = pd.read_csv(M3_MINUS_M0_ZSCORES_RES_ON_SITE)
    df["response"] = df["response"].replace({"GR": 1, "PaR": 0, "NR": 0})

    # ===== residualize on site
    rois = [r for r in list(df.columns) if r.endswith("_GM_Vol") or r.endswith("_CSF_Vol")]
    residualizer = Residualizer(data=df[["age","sex","site","response"]], formula_res="site", formula_full="site + sex + age + response")
    Zres = residualizer.get_design_mat(df[["age","sex","site","response"]])
    X= df[rois].values
    residualizer.fit(X, Zres)
    X = residualizer.transform(X, Zres)
    df[rois]=X
    # ===== end residualize on site

    # for r in HYP_AMYG_GM_ROIS:
    #     num_positive = (df[r] > 0).sum()
    #     print(r, "  ",num_positive)

    # center age
    df["age"] = df["age"] - df["age"].mean()
    print(df["site"].value_counts())

    df, string_dict = transform_df(df)
    rois = [r for r in list(df.columns) if r.endswith("_GM_Vol")]# or r.endswith("_CSF_Vol")]
    
    rows = []
    val_of_interest = "Intercept" if intercept else "C(response)[T.1]"
    # df contains rois, participant_id, age, sex, and site columns
    for col in rois:
        formula = f"{col} ~ age + C(sex)" if intercept else f"{col} ~ age + C(sex) + C(site) + C(response)"
        model = smf.ols(formula, data=df).fit()
        roi_name = string_dict[col]
        pval = model.pvalues[val_of_interest] 
        tstat = model.tvalues[val_of_interest] 
        # if model.pvalues['C(sex)[T.male]']<=0.05: print(col, " sex pval: ",model.pvalues['C(sex)[T.male]'])
        # if model.pvalues['age']<=0.05 : print(col ," age pval:",model.pvalues['age'])

        rows.append({"ROI": roi_name,
                        "pvalue": pval,
                        "tstatistic": tstat})

    results_df = pd.DataFrame(rows, columns=["ROI", "pvalue", "tstatistic"])
    print(results_df)

    # print only significant rows
    significant = results_df[results_df["pvalue"] <= 0.05]
    significant_sorted = significant.sort_values(by="pvalue", ascending=True)

    # with pd.option_context('display.max_rows', None):
    print("\nSignificant (all rows, sorted by p-value):\n", significant_sorted)
    print(len(significant_sorted))

    print("\n",results_df[results_df["ROI"].isin(HYP_AMYG_GM_ROIS)])
    print("\n",results_df[results_df["ROI"].isin(HYP_AMYG_CSF_ROIS)])

    return results_df


"""
pvalues and tstatstics associated with intercept
without C(site)
                                                   ROI        pvalue  tstatistic
81   Left MPrG precentral gyrus medial segment_GM_Vol  2.321634e-10    7.159929
244                Right PrG precentral gyrus_CSF_Vol  3.291939e-10   -7.084048
206            Right MFG middle frontal gyrus_CSF_Vol  9.543377e-10   -6.851520
246                 Right PT planum temporale_CSF_Vol  1.012193e-09   -6.838603
104                Right PoG postcentral gyrus_GM_Vol  1.697781e-09    6.724808
..                                                ...           ...         ...
4                               Right Amygdala_GM_Vol  3.663023e-02    2.122208
86                    Right OCP occipital pole_GM_Vol  3.867904e-02    2.099027
168                      Left Basal Forebrain_CSF_Vol  3.975626e-02   -2.087262
221                   Left OCP occipital pole_CSF_Vol  4.283900e-02   -2.055051
76             Right MOrG medial orbital gyrus_GM_Vol  4.362134e-02    2.047194

[231 rows x 3 columns]

                          ROI    pvalue  tstatistic
4      Right Amygdala_GM_Vol  0.036630    2.122208
5       Left Amygdala_GM_Vol  0.240825    1.180900
16  Right Hippocampus_GM_Vol  0.000006    4.808216
17   Left Hippocampus_GM_Vol  0.000022    4.487220


with C(site)
                                                    ROI    pvalue  tstatistic
83   Left MSFG superior frontal gyrus medial segmen...  0.000002    5.123255
253        Left SMC supplementary motor cortex_CSF_Vol  0.000010   -4.747922
252       Right SMC supplementary motor cortex_CSF_Vol  0.000027   -4.484078
225  Left OpIFG opercular part of the inferior fron...  0.000057   -4.276684
250           Right SFG superior frontal gyrus_CSF_Vol  0.000082   -4.173187
..                                                 ...       ...         ...
234                Right PIns posterior insula_CSF_Vol  0.038583   -2.106676
247                   Left PT planum temporale_CSF_Vol  0.041188   -2.078402
190                   Right FuG fusiform gyrus_CSF_Vol  0.042013   -2.069775
105                  Left PoG postcentral gyrus_GM_Vol  0.046805    2.022343
178                Right Calc calcarine cortex_CSF_Vol  0.049774   -1.995021

[114 rows x 3 columns]

                          ROI    pvalue  tstatistic
4      Right Amygdala_GM_Vol  0.668241    0.430306
5       Left Amygdala_GM_Vol  0.548016    0.603546
16  Right Hippocampus_GM_Vol  0.153960    1.440646
17   Left Hippocampus_GM_Vol  0.121282    1.567672


pvalues and tstatstics associated with response
                                                 ROI    pvalue  tstatistic
5                              Left Amygdala_GM_Vol  0.049848    1.994827
10                  Left Cerebellum Exterior_GM_Vol  0.037707    2.117101
12              Left Cerebellum White Matter_GM_Vol  0.036421   -2.131977
28                          Right Ventral DC_GM_Vol  0.029757    2.217343
51                  Left Ent entorhinal area_GM_Vol  0.024436    2.298560
59                     Left GRe gyrus rectus_GM_Vol  0.033753    2.164372
132      Right TTG transverse temporal gyrus_GM_Vol  0.045843   -2.032012
138                          Right Amygdala_CSF_Vol  0.037210    2.122801
166  Cerebellar Vermal Lobules VI_MINUS_VII_CSF_Vol  0.035844   -2.138810

                          ROI    pvalue  tstatistic
4      Right Amygdala_GM_Vol  0.470735    0.725110
5       Left Amygdala_GM_Vol  0.049848    1.994827
16  Right Hippocampus_GM_Vol  0.381808    0.879961
17   Left Hippocampus_GM_Vol  0.227040    1.218424

"""

def print_roi_names_long(stats_df, p_col, df_str, t_col):
    atlas_df = pd.read_csv(ROOT+"data/processed/lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=';')
    roi_names_map = dict(zip(atlas_df['ROI_Neuromorphometrics_labels'], atlas_df['ROIname']))
    stats_df.ROI = [roi_names_map[val] for val in list(stats_df["ROI"])]
    print(stats_df[["ROI",p_col, f"{df_str}_p",t_col]])
    
def get_glass_brain_t_statistics(res="no_res", m0=False, m3minusm0=False, intercept_analysis=False, \
                  WM_roi=False, only_significant_roi=False, four_rois=False):
    
    str_WM = "_WM_Vol" if WM_roi else ""
    df_str = "intercept" if intercept_analysis else "response"
    file_name = "statsuniv_rois"
    if only_significant_roi: file_name = file_name + "_significant_roi"
    if m3minusm0:
        file_name = "m3_minus_m0_" + file_name
        if intercept_analysis:
                file_name += "_intercept_analysis_of_changem3m0"

    stats_df = pd.read_excel(STATS_UNIV_RES_DIR + file_name + str_WM + ".xlsx")
    t_col = f"{df_str}_t"
    p_col = f"{df_str}_pcor_bonferroni" if (only_significant_roi and m0) or intercept_analysis else f"{df_str}_p"
    print(len(stats_df[["ROI",p_col, f"{df_str}_p",t_col]]))
    print("statistics in order from lowest to highest corrected pvalue:\n", stats_df[["ROI",p_col,t_col]].sort_values(by=p_col))
    stats_df_four_rois = stats_df[stats_df["ROI"].isin(["Right Hippocampus_GM_Vol", "Left Hippocampus_GM_Vol",\
                                         "Right Amygdala_GM_Vol", "Left Amygdala_GM_Vol"])][["ROI",p_col,t_col]]
    print("\nbilateral amygdala and hippocampus statistics :\n", stats_df_four_rois)

    stats_df = stats_df[stats_df[p_col]<0.05]
    print("\n\nROI with significative pvalues:\n",stats_df[["ROI",p_col, f"{df_str}_p",t_col]])
    
    # other_df = pd.read_excel("reports/stats_univ_results/statsuniv_rois.xlsx")
    # print(other_df[other_df["ROI"].isin(stats_df["ROI"].values)][["ROI",p_col]])
    # print_roi_names_long(stats_df, p_col,df_str, t_col)
    if four_rois: stats_dict_tstat= {k:v for (k,v) in zip(stats_df_four_rois["ROI"].values, stats_df_four_rois[t_col].values)}
    else : stats_dict_tstat= {k:v for (k,v) in zip(stats_df["ROI"].values, stats_df[t_col].values)}
    if not WM_roi : 
        if not only_significant_roi and m3minusm0:
            # in this case there are too many significant ROI and there's overlap btw the GM and CSF volumes
            stats_dict_tstatGM = {k: v for k, v in stats_dict_tstat.items() if k.endswith("_GM_Vol")}
            stats_dict_tstatCSF = {k: v for k, v in stats_dict_tstat.items() if k.endswith("_CSF_Vol")}
        else : 
            stats_dict_tstat = {k: v if k.endswith("_GM_Vol") else -v for k, v in stats_dict_tstat.items()}
    str_corr = "Bonferroni corrected" if (only_significant_roi and m0) or intercept_analysis else "uncorrected"
    m_str = "m0" if m0 else "m3-m0"
    if intercept_analysis: title = str_corr+" t-statistics of significant ROI associated with anatomical change : "+m_str+" ~ 1 + age + sex + site"
    else : 
        title = str_corr+" t-statistics of significant ROI associated with Li response : "+m_str+" ~ resp Li + age + sex + site"
    if four_rois and not intercept_analysis: 
        title = "t-statistics of bilateral hippocampus and amygdala associated with Li response : "+m_str+" ~ resp Li + age + sex + site"

    if not only_significant_roi and m3minusm0 and intercept_analysis: 
        if WM_roi: plot_glassbrain(dict_plot=stats_dict_tstat, title=title+ " WM")
        else: 
            plot_glassbrain(dict_plot=stats_dict_tstatGM, title=title+ " GM")
            plot_glassbrain(dict_plot=stats_dict_tstatCSF, title=title+ " CSF")
    else : plot_glassbrain(dict_plot=stats_dict_tstat, title=title)



def main():
    # mean_roi_vals_hip_amyg()
    test_M0()
    quit()
    """
    M3-M0 ~ resp, age, sex, site
    M3-M0 ~ 1, CS(age), sex , site 
    faire att à bien encode sex et site comme categorical variables
    no residualization
    
    """
    test_M3minusM0()
 


if __name__ == "__main__":
    main()


