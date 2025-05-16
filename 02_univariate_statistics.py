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

# inputs
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
VBMLOOKUP_FILE = "/drf/local/spm12/tpm/labels_Neuromorphometrics.xml"
ROOT = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR = ROOT+"data/processed/"
FEAT_IMPTCE_RES_DIR = ROOT+"reports/feature_importance_results/"

# outputs
STATS_UNIV_RES_DIR = ROOT+"reports/stats_univ_results/"


def m3minusm0(save_m3_minus_m0_df=False,WM_roi = False):
    """
    Saves a df of M3-M0 ROI measures
        save_m3_minus_m0_df (bool) : if True, save df of differences between m3 and m0 to csv. if False, don't. (no standard scaling at this point) 
        WM_roi (bool) : white matter volumes only
    """

    if WM_roi : path_df_M3minusM0 = DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site_WM_Vol.csv"
    else : path_df_M3minusM0 = DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site.csv"

    if not os.path.exists(path_df_M3minusM0):
        if WM_roi: df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03_WM_Vol.csv")
        else : df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03.csv")
        dfROIM00 = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M00"].reset_index(drop=True)
        dfROIM03 = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M03"].reset_index(drop=True)

        # Merge M03 and M00 on participant_id
        merged = pd.merge(dfROIM03, dfROIM00, on="participant_id", suffixes=("_M03", "_M00"))

        list_ = ["y","age","sex","site"]
        for l in list_:
            assert (merged[l+'_M00'] == merged[l+'_M03']).all(), " issue with "+l+" between same subjects at M00 and M03"

        columns_M03 = [col for col in merged.columns if col.endswith("_M03")]
        columns_M00 = [col for col in merged.columns if col.endswith("_M00")]
        common_columns = [col[:-4] for col in columns_M03 if col[:-4] + "_M00" in columns_M00]
        print("common_columns ", len(common_columns)) # == 273 since there are 268 ROIs (134 GM and 134 CSF) and y (label), age, sex, site, session
        print(merged)

        # creating a df for differences of M03-M00
        differences_df = pd.DataFrame({
            col: merged[col + "_M03"] - merged[col + "_M00"] for col in common_columns if not col in ["age","sex","site","y","session"]
        })

        differences_df.insert(0, 'participant_id', merged['participant_id'])
        differences_df["y"] = merged["y_M00"]
        differences_df["age"] = merged["age_M00"]
        differences_df["sex"] = merged["sex_M00"]
        differences_df["site"] = merged["site_M00"]

        print(differences_df)

        if save_m3_minus_m0_df : differences_df.to_csv(path_df_M3minusM0,index=False)  

    else : differences_df = pd.read_csv(path_df_M3minusM0)

    print("differences grouped by label\n",differences_df[get_rois(WM=WM_roi)+["y"]].groupby("y").median())
    differences_df = differences_df.drop("participant_id", axis=1)
    print(differences_df)



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

def westfall_young_maxT(df_X, df_str, y_str, covariates, formula_base,significant_rois, B=1000, seed=42):
    np.random.seed(seed)

    # transform df column names (as described in string_dict) to fit statsmodels requirements
    if significant_rois: df_X = df_X[significant_rois + ["age", "sex", "site", "y"]]
    df_X, string_dict = transform_df(df_X)
    list_rois = [col for col in df_X.columns if col not in ["age", "sex", "site", "y"]]
    stats = []
    if y_str=="Intercept":
        df_X_h1 = df_X.copy()
        print("Scaling covariates for intercept analysis...")
        for col in ["age", "sex", "site"]:
            df_X_h1[col] = (df_X_h1[col] - df_X_h1[col].mean()) / df_X[col].std()

    # fit OLS models and store original t-statistics for 'resp'
    for roi in list_rois:
        model = smf.ols(f"{roi} ~ {formula_base}", df_X_h1).fit()
        tvals = model.tvalues[[y_str] + covariates].tolist()
        pvals = model.pvalues[[y_str] + covariates].tolist()
        stats.append([roi] + tvals + pvals)
    
    cols = ['ROI'] + \
           [f"{df_str}_t"] + [f"{c}_t" for c in covariates] + \
           [f"{df_str}_p"] + [f"{c}_p" for c in covariates]
    stats_df = pd.DataFrame(stats, columns=cols)
    h1_tstats_dict = dict(zip(stats_df["ROI"], stats_df[f"{df_str}_t"]))
    
    # permutation loop
    max_tstats = []
    for i in range(B):
        df_perm = df_X.copy()
        # permuting the response labels if looking at the pvalues of response to Li
        # permuting the m3-m0 roi values if looking at the pvalues of the intercept
        if y_str=="y": df_perm[y_str] = np.random.permutation(df_perm[y_str])
        elif y_str =="Intercept": 
            df_perm[list_rois] = df_perm[list_rois].sample(frac=1, replace=False).reset_index(drop=True)
            print("Scaling covariates for intercept analysis for permuation "+str(i)+"...")
            for col in ["age", "sex", "site"]:
                # permutation of rois within each row (each subject)
                df_perm[col] = (df_perm[col] - df_perm[col].mean()) / df_perm[col].std()
        tstats_perm = []

        for roi in list_rois:
            # for each permutation, for each roi, fill tstats_perm with the list of absolute t-statistics values
            model = smf.ols(f"{roi} ~ {formula_base}", df_perm).fit()
            tstats_perm.append(np.abs(model.tvalues[y_str]))
            # for each permutation, fill the list max_tstats with the maximum value of the absolute values of t-statistics across all ROIs
        max_tstats.append(np.max(tstats_perm))
    
    # h0 t-stats
    max_tstats = np.array(max_tstats)

    # compute adjusted p-values
    corrected_pvals = {
        roi: np.mean(max_tstats > np.abs(h1_tstats_dict[roi]))
        for roi in list_rois
    }

    stats_df[f"{df_str}_pcor_Tmax"] = list(corrected_pvals.values())
    stats_df["ROI"] = stats_df['ROI'].replace(string_dict)

    return stats_df[["ROI",f"{df_str}_pcor_Tmax"]]




def perform_tests(res="no_res", save=False, m0=False, m3minusm0=False, include_site=True, intercept_analysis=False, \
                  WM_roi=False, only_significant_roi=False, westfall_and_young =False):
    """
    Perform univariate OLS regressions on brain ROI volumes with optional residualization and model variations.

        res (str) "no_res", "res_age_sex" or "res_age_sex_site" : type of residualization applied to ROI before statistical testing
        save (bool) : save the results in pkl file
        m0 (bool): perform statistical tests on m0 ROI
        m3minusm0 (bool) : perform statistical tests on the difference of m3 ROI and m0 ROI for all subjects with MRI scans before and after Li intake.
        include_site (bool) : if we include site in the OLS regression (if False, OLS is fitted on Li response, sex, and age; if True, OLS
                                is fitted on the same variables + acquisition site)
        intercept_analysis (bool) : if False, we look at the effect of Lithium on either m0 or m3-m0 ROI values. if True,
                                    we look at the effect of the intercept on m3-m0 values (or whaterver "dataframe" contains). 
                                    This means that we look for a statistically significant differences between m3-m0 and 0, which is equivalent
                                    to looking for a statiscally significant change in ROI values between before and after Li intake.  
        WM_roi (bool) : if we choose to perform the tests on white matter volumes only
        only_significant_roi (bool) : if we choose to perform the tests on signficiant ROI as found using a feature importance method (here, SHAP values)
    """

    assert res in ["res_age_sex_site", "res_age_sex", "no_res"], f"Invalid residualization type: {res}"

    if m0 and intercept_analysis:
        raise ValueError("Intercept analysis is only relevant for m3-m0 (analyzing change before/after Li intake), not for m0 values.")
    
    str_WM = "_WM_Vol" if WM_roi else ""
    df_file = None

    if m0:
        df_file = f"{DATA_DIR}df_ROI_age_sex_site_M00{str_WM}.csv"
    elif m3minusm0:
        df_file = f"{DATA_DIR}df_ROI_M03_minus_M00_age_sex_site{str_WM}.csv"
    else:
        raise ValueError("You must select either m0=True or m3minusm0=True")

    df_X = pd.read_csv(df_file)
    significant_rois=None

    if only_significant_roi and not WM_roi:
        significant_df = pd.read_excel(FEAT_IMPTCE_RES_DIR+"significant_shap_mean_abs_value_pvalues_1000_random_permut.xlsx")
        significant_rois = [roi for roi in list(significant_df.columns) if roi!="fold"]
        df_X = df_X[significant_rois + ["age", "sex", "site", "y"]]
    else: df_X = df_X[get_rois(WM=WM_roi) + ["age", "sex", "site", "y"]]
    
    df_X["y"] = df_X["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    
    if intercept_analysis:
        print("Scaling covariates for intercept analysis...")
        for col in ["age", "sex", "site"]:
            df_X[col] = (df_X[col] - df_X[col].mean()) / df_X[col].std()

    # transform df column names (as described in string_dict) to fit statsmodels requirements
    df_X, string_dict = transform_df(df_X)

    roi_columns = [col for col in df_X.columns if col not in ["age", "sex", "site", "y"]]
    if only_significant_roi: assert len(roi_columns) == 17, f"Unexpected number of ROIs: {len(roi_columns)}"
    else:
        if WM_roi : assert len(roi_columns) == 134, f"Unexpected number of ROIs: {len(roi_columns)}"
        else : assert len(roi_columns) == 268, f"Unexpected number of ROIs: {len(roi_columns)}"

    stats = []
    y_str = "Intercept" if intercept_analysis else "y"
    df_str = "intercept" if intercept_analysis else "response"
    covariates = ["sex", "age"] + (["site"] if include_site else [])
    formula_base = " + ".join(["1"] + covariates) if intercept_analysis else " + ".join(["y"] + covariates)
    
    for roi in roi_columns:
        model = smf.ols(f"{roi} ~ {formula_base}", df_X).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        # print(model.model.data.param_names) # if intercept_analysis False : \
        # outputs : ['Intercept', 'y', 'sex', 'age', 'site'], if intercep_analysis True, \
        # outputs : ["Intercept","sex","age","site"]

        tvals = model.tvalues[[y_str] + covariates].tolist()
        pvals = model.pvalues[[y_str] + covariates].tolist()

        aov_stats = []
        if not intercept_analysis:
            aov_stats += [aov.loc[y_str, "F"], aov.loc[y_str, "PR(>F)"]]
        for cov in covariates:
            aov_stats += [aov.loc[cov, "F"], aov.loc[cov, "PR(>F)"]]

        stats.append([roi] + tvals + pvals + aov_stats)

    # Build column names dynamically
    cols = ['ROI'] + \
           [f"{df_str}_t"] + [f"{c}_t" for c in covariates] + \
           [f"{df_str}_p"] + [f"{c}_p" for c in covariates]
    if not intercept_analysis:
        cols += [f"{df_str}_f", f"{df_str}_p_anova"]
    cols += [f"{c}_f_anova" for c in covariates] + [f"{c}_p_anova" for c in covariates]

    stats_df = pd.DataFrame(stats, columns=cols)
    stats_df['ROI'] = stats_df['ROI'].replace(string_dict)
    df_X.rename(columns=string_dict, inplace=True)

    # Multiple corrections
    target_p = stats_df[f"{df_str}_p"].values
    _, pcor_fdr_bh, _, _ = multipletests(target_p, method='fdr_bh')
    _, pcor_bonf, _, _ = multipletests(target_p, method='bonferroni')
    stats_df[f"{df_str}_pcor_fdr_bh"] = pcor_fdr_bh
    stats_df[f"{df_str}_pcor_bonferroni"] = pcor_bonf

    df_X_original = pd.read_csv(df_file)
    if westfall_and_young: t_max_pvals_df = westfall_young_maxT(df_X_original, df_str, y_str, covariates, formula_base, significant_rois)

    if not intercept_analysis:
        target_p_anova = stats_df[f"{df_str}_p_anova"].values
        _, pcor_fdr_bh_anova, _, _ = multipletests(target_p_anova, method='fdr_bh')
        _, pcor_bonf_anova, _, _ = multipletests(target_p_anova, method='bonferroni')
        stats_df[f"{df_str}_pcor_fdr_bh_anova"] = pcor_fdr_bh_anova
        stats_df[f"{df_str}_pcor_bonferroni_anova"] = pcor_bonf_anova

    # Print summary
    print(stats_df)
    print(f"\nNumber of ROI with {df_str} p < 0.05:", (target_p < 0.05).sum())
    print(f"Bonferroni corrected p < 0.05:", (pcor_bonf < 0.05).sum())
    print(f"FDR corrected p < 0.05:", (pcor_fdr_bh < 0.05).sum())

    t_col = f"{df_str}_t"
    p_col = f"{df_str}_pcor_bonferroni"
    # print(stats_df.loc[stats_df[f"{df_str}_p"] < 0.05][["ROI", f"{df_str}_p",t_col, f"{df_str}_p_anova", f"{df_str}_pcor_bonferroni"]])

    filtered_sorted_stats = stats_df.loc[stats_df[p_col] < 0.05].copy()
    filtered_sorted_stats["abs_t"] = filtered_sorted_stats[t_col].abs()
    filtered_sorted_stats = filtered_sorted_stats.sort_values(by="abs_t", ascending=False)

    if westfall_and_young:  
        stats_df_with_tmax = pd.merge(stats_df,t_max_pvals_df, on="ROI", how="inner" )
        print(stats_df_with_tmax)
        print(f"Westfall and Young corrected p < 0.05:", (stats_df_with_tmax[f"{df_str}_pcor_Tmax"] < 0.05).sum())
        print("roi with Tmax corrected pvalue <0.05 : \n",\
            stats_df_with_tmax[stats_df_with_tmax[f"{df_str}_pcor_Tmax"] < 0.05][["ROI",f"{df_str}_t",f"{df_str}_p",f"{df_str}_pcor_Tmax"]])

    quit()

    if not intercept_analysis:
        print("in order of highest to lowest absolute t statistic, the ROI with bonferroni corrected pvalues <0.05 are:\n",\
              filtered_sorted_stats[["ROI", f"{df_str}_p",t_col, f"{df_str}_p_anova", f"{df_str}_pcor_bonferroni", f"{df_str}_pcor_bonferroni_anova"]])
    else:
        print("in order of highest to lowest absolute t statistic, the ROI with bonferroni corrected pvalues <0.05 are:\n",\
              filtered_sorted_stats[["ROI", f"{df_str}_p",t_col,  p_col]])

    if save:
        save_name = "statsuniv_rois"
        if only_significant_roi: save_name = save_name + "_significant_roi"
        if m3minusm0:
            save_name = "m3_minus_m0_" + save_name
            if intercept_analysis:
                save_name += "_intercept_analysis_of_changem3m0"
        if westfall_and_young: stats_df_with_tmax.to_excel(STATS_UNIV_RES_DIR + save_name + str_WM + ".xlsx", index=False)
        else : stats_df.to_excel(STATS_UNIV_RES_DIR + save_name + str_WM + ".xlsx", index=False)

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
    if not only_significant_roi and m3minusm0:
        # in this case there are too many significant ROI and there's overlap btw the GM and CSF volumes
        stats_dict_tstatGM = {k: v for k, v in stats_dict_tstat.items() if k.endswith("_GM_Vol")}
        stats_dict_tstatCSF = {k: v for k, v in stats_dict_tstat.items() if k.endswith("_CSF_Vol")}
    else : 
        stats_dict_tstat = {k: v if k.endswith("_GM_Vol") else -v for k, v in stats_dict_tstat.items()}
        print(stats_dict_tstat)
    str_corr = "Bonferroni corrected" if (only_significant_roi and m0) or intercept_analysis else "uncorrected"
    m_str = "m0" if m0 else "m3-m0"
    if intercept_analysis: title = str_corr+" t-statistics of significant ROI associated with anatomical change : "+m_str+" ~ 1 + age + sex + site"
    else : 
        title = str_corr+" t-statistics of significant ROI associated with Li response : "+m_str+" ~ resp Li + age + sex + site"
    if four_rois and not intercept_analysis: 
        title = "t-statistics of bilateral hippocampus and amygdala associated with Li response : "+m_str+" ~ resp Li + age + sex + site"

    if not only_significant_roi and m3minusm0 and intercept_analysis: 
        plot_glassbrain(dict_plot=stats_dict_tstatGM, title=title+ " GM")
        plot_glassbrain(dict_plot=stats_dict_tstatCSF, title=title+ " CSF")
    else : plot_glassbrain(dict_plot=stats_dict_tstat, title=title)



def main():

    get_glass_brain_t_statistics(res="no_res", m0=True, m3minusm0=False, intercept_analysis=False, \
                  WM_roi=False, only_significant_roi=True,  four_rois=False)

    

    perform_tests(res="no_res", save=False, m0=False, m3minusm0=True, include_site=True, intercept_analysis=True, WM_roi=False, only_significant_roi=True)

    # perform_tests(res="no_res", save=False, m0=True, m3minusm0=False, include_site=True, intercept_analysis=False, WM_roi=True)
    quit()
    # perform_tests(save=False)    
    # quit()
    m3minusm0(save_m3_minus_m0_df=True,  save_stats_df=True) 


if __name__ == "__main__":
    main()


