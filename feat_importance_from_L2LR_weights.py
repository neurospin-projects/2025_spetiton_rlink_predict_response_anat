from utils import read_pkl, get_rois, save_pkl
import numpy as np, os, pandas as pd

# inputs
WEIGHTS_PATH = "reports/classification_results/coefficientsL2LR/"
CLASSIF_RES = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/reports/classification_results/"
BOOTSTRAP_REG = CLASSIF_RES+"results_classification_GRvsPaRNR_1000fold_v4labels_24juin25_42cvseed.pkl"
BOOSTRAP_WHITENED = CLASSIF_RES+"results_classification_GRvsPaRNR_1000fold_v4labels_whitened_24juin25_42cvseed.pkl"
WEIGHTS_BOOT_REG = CLASSIF_RES+"coefficientsL2LR/L2LR_coefficients_GRvsPaRNR_1000fold_v4labels_24juin25_42cvseed.pkl"
WIEGHTS_BOOT_WHITENED = CLASSIF_RES+"coefficientsL2LR/L2LR_coefficients_GRvsPaRNR_1000fold_v4labels_whitened_24juin25_42cvseed.pkl"
#outputs
RES_PATH = "reports/feature_importance_results/mean_abs_L2LRcoeffs_whitened_and_regular.pkl"

boot_reg = read_pkl(BOOTSTRAP_REG)
print(boot_reg["roc_auc_test"].mean())
boot_reg_w = read_pkl(WEIGHTS_BOOT_REG)
print(boot_reg_w)


weights_matrix = np.vstack(boot_reg_w['res_age_sex_site'].values)  # shape: (n_bootstraps, n_features)
weights_df = pd.DataFrame(weights_matrix, columns=get_rois())
coefs_stat = weights_df.describe(percentiles=[.975, .5, .025])
print(np.shape(weights_matrix))
print(weights_df)
print(coefs_stat[["Right Amygdala_GM_Vol", "Left Amygdala_GM_Vol"]])
print(weights_df[["Right Amygdala_GM_Vol", "Left Amygdala_GM_Vol"]])


# Get the 2.5th and 97.5th percentiles
lower_bound = coefs_stat.loc['2.5%']
upper_bound = coefs_stat.loc['97.5%']

print(lower_bound)
print(upper_bound)

# Identify columns where 0 is outside the confidence interval (the ROI that with significant coefficients in the linear model)
columns_no_zero_in_CI = coefs_stat.columns[(lower_bound > 0) | (upper_bound < 0)].tolist()
print(columns_no_zero_in_CI)
print(len(columns_no_zero_in_CI))

df = weights_df[columns_no_zero_in_CI].copy()
print(df)

# invert CSF 
list_cols = list(df.columns)
df.loc[:, df.columns.str.endswith("_CSF_Vol")] *= -1

"""
20 columns : from weights in the latent (whitened) space

['4th Ventricle_GM_Vol', 'Left Accumbens Area_GM_Vol', 'Right Amygdala_GM_Vol', 'Left Cerebellum White Matter_GM_Vol', 
'Right Hippocampus_GM_Vol', 'Right Ventral DC_GM_Vol', 'Cerebellar Vermal Lobules I-V_GM_Vol', 'Right FO frontal operculum_GM_Vol',
 'Left MTG middle temporal gyrus_GM_Vol', 'Right OpIFG opercular part of the inferior frontal gyrus_GM_Vol', '4th Ventricle_CSF_Vol',
   'Right Thalamus Proper_CSF_Vol', 'Left Basal Forebrain_CSF_Vol', 'Left FO frontal operculum_CSF_Vol', 'Left MFG middle frontal gyrus_CSF_Vol',
     'Left OpIFG opercular part of the inferior frontal gyrus_CSF_Vol', 'Right PCu precuneus_CSF_Vol', 
'Left PrG precentral gyrus_CSF_Vol', 'Left SOG superior occipital gyrus_CSF_Vol', 'Right TMP temporal pole_CSF_Vol']

"""


"""
if not os.path.exists(RES_PATH):
    res = []
    whitened_path = WEIGHTS_PATH+"L2LR_coefficients_GRvsPaRNR_5fold_v4labels_whitened"\
            +"_24juin25_42cvseed.pkl"
    regular_path = WEIGHTS_PATH+"L2LR_coefficients_GRvsPaRNR_5fold_v4labels"\
            +"_24juin25_42cvseed.pkl"

    data_whitened_true_labels = read_pkl(whitened_path)
    data_reg_true_labels = read_pkl(regular_path)
    print(data_whitened_true_labels)
    print(data_reg_true_labels)

    weights_whitened = np.stack(data_whitened_true_labels["res_age_sex_site"].values) # shape (5, 268)
    mean_weights_whitened = np.mean(np.abs(weights_whitened),axis=0) # shape (268,)

    weights_reg = np.stack(data_reg_true_labels["res_age_sex_site"].values) # shape (5, 268)
    mean_weights_reg = np.mean(np.abs(weights_reg),axis=0) # shape (268,)
    res.append({"permutation_labels_seed":0,"mean_weights":mean_weights_whitened ,"whitened":True})
    res.append({"permutation_labels_seed":0,"mean_weights":mean_weights_reg ,"whitened":False})


    for seed in range(1,1001):
        whitened_path = WEIGHTS_PATH+"L2LR_coefficients_GRvsPaRNR_5fold_v4labels_whitened_random_permutations_with_seed_"+str(seed)\
            +"_24juin25_42cvseed.pkl"
        regular_path = WEIGHTS_PATH+"L2LR_coefficients_GRvsPaRNR_5fold_v4labels_random_permutations_with_seed_"+str(seed)\
            +"_24juin25_42cvseed.pkl"
        data_whitened = read_pkl(whitened_path)
        data_reg = read_pkl(regular_path)

        weights_whitened = np.stack(data_whitened["res_age_sex_site"].values) # shape (5, 268)
        mean_weights_whitened = np.mean(np.abs(weights_whitened),axis=0) # shape (268,)

        weights_reg = np.stack(data_reg["res_age_sex_site"].values) # shape (5, 268)
        mean_weights_reg = np.mean(np.abs(weights_reg),axis=0) # shape (268,)

        print(mean_weights_whitened.shape)
        print(data_reg["res_age_sex_site"])
        res.append({"permutation_labels_seed":seed,"mean_weights":mean_weights_whitened ,"whitened":True})
        res.append({"permutation_labels_seed":seed,"mean_weights":mean_weights_reg ,"whitened":False})

    res_df = pd.DataFrame(res)
    print(res_df)
    # res_df.to_csv(RES_PATH,index=False)
    save_pkl(res_df, RES_PATH)
    print("dataframe saved to ", RES_PATH)

else:
    res_df = read_pkl(RES_PATH)
    print(res_df)
    roi_names= get_rois()
    alpha= 0.05
    res_df_reg = res_df[res_df["whitened"]==True]
    print(res_df_reg)
    res_df_reg_expanded = pd.DataFrame(res_df_reg["mean_weights"].tolist(), columns=roi_names)
    res_df_reg_expanded.insert(0, "fold", res_df_reg["permutation_labels_seed"].values) # add fold col back
    print(res_df_reg_expanded)
    # res_df_reg_expanded = res_df_reg_expanded.head(50)

    # Extract weights values for the true model (h1 = fold 0)
    row_h1_values = res_df_reg_expanded[res_df_reg_expanded["fold"] == 0].iloc[0]
    print(row_h1_values)
    roi_weights_h1 = row_h1_values[roi_names]

    # Sort them in descending order
    sorted_roi_weights = roi_weights_h1.sort_values(ascending=False)

    # Display top N ROIs
    top_n = 40
    print(sorted_roi_weights.head(top_n))

    quit()

    # --- Uncorrected p-values (per ROI) ---
    # Compare weights from permutations (h0) to the h1 weights values
    comparison_df = res_df_reg_expanded[res_df_reg_expanded["fold"] != 0][roi_names] > row_h1_values[roi_names]
    # count for each ROI (column) how many rows had higher values than the first row
    # divide by number of rows of comparison_df -1 (nb folds without counting h1)
    pvalues = comparison_df.sum(axis=0) / (len(comparison_df) - 1)

    # Append uncorrected p-values as a new row
    pvalues_row = pvalues.to_frame().T  # convert to single-row DataFrame
    pvalues_row.insert(0, "fold", "pvalues") # add "pvalues" to "fold" column
    res_df_reg_expanded = pd.concat([res_df_reg_expanded, pvalues_row], ignore_index=True)

    print(res_df_reg_expanded[["fold","Left Amygdala_GM_Vol","Right Amygdala_GM_Vol", "Left Hippocampus_GM_Vol","Right Hippocampus_GM_Vol"]])
    # pvalues for left amygdala, right amygdala, left hippocampus and right hippocampus:
    # 0.031031               0.176176                 0.069069                  0.545546

    # --- Compute max weights across ROIs for each permutation (needed for maxT correction) ---
    # Skip the h1 row (fold 0), only consider folds 1 to 1000
    res_df_reg_expanded["row_max"] = res_df_reg_expanded[roi_names].max(axis=1)
    res_df_reg_expanded.loc[res_df_reg_expanded["fold"] == "pvalues", "row_max"] # Don't use pvalues row
    res_df_reg_expanded.loc[res_df_reg_expanded["fold"] == 0, "row_max"] = np.nan # Don't use h1 row

    row_max_values = res_df_reg_expanded.loc[1:1000, "row_max"].values  # max across ROIs for each permutation

    # --- Westfall & Young maxT-corrected p-values ---
    corrected_pvals = []
    for roi in roi_names:
        h1_val = row_h1_values[roi]
        # Count how many times the max weights across ROIs (under permutations) exceeds h1 weights for this ROI
        p_corr = (row_max_values > h1_val).sum() / len(row_max_values)
        corrected_pvals.append(p_corr)

    # Add corrected p-values row
    pvalues_corrected = ["corrected_pvalues"] + corrected_pvals + [np.nan]
    pvalues_corrected_df = pd.DataFrame([pvalues_corrected], columns=res_df_reg_expanded.columns)
    res_df_reg_expanded = pd.concat([res_df_reg_expanded, pvalues_corrected_df], ignore_index=True)

    # --- Extract and report significant ROIs ---
    pvalues_row = res_df_reg_expanded[res_df_reg_expanded["fold"] == "pvalues"].iloc[0]
    pvalues_corrected_row = res_df_reg_expanded[res_df_reg_expanded["fold"] == "corrected_pvalues"].iloc[0]

    uncorrected_significant_roi = [roi for roi in roi_names if pvalues_row[roi] < alpha]
    corrected_significant_roi = [roi for roi in roi_names if pvalues_corrected_row[roi] < alpha]

    print("Uncorrected significant ROIs:", uncorrected_significant_roi, 
        f"\nTotal: {len(uncorrected_significant_roi)}")

    print("\nCorrected significant ROIs:", corrected_significant_roi, 
        f"\nTotal: {len(corrected_significant_roi)}")
"""