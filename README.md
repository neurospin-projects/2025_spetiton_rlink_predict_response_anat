# 2025_spetiton_rlink_predict_response_anat
R-LiNK Predict Li response with Anat MRI


# datasets
01_create_dataframe_ROIs.py creates the dataframes of ROIs of GM volume and CSF volume with age, sex, site, and participant_id for all participants.
default is M00 measures only (before Li intake) and GM and CSF measures (268 ROI total, 134 for GM, 134 for CSF).
At M00, N = 116.
with WM=True in save_df_ROI(), it saves the df of ROIs of WM volume with age, sex, site, and participant_id for all participants
with longitudinal=True in save_df_ROI(), it saves the df of ROIs (eitehr GM and CSF or WM volumes) for subjects with both M00 and M03 measures (N=91).

# classification

# feature importance analyses (using SHAP values)
Results in df saved at SHAP_summary_res_age_sex_site_h0h1_1000_random_permut.xlsx (or .pkl): two columns: one “fold” where fold==0 refers to the SHAP values computed with true labels, and where fold ==1 to fold ==1000 are the SHAP values computed with randomly permuted labels; “mean_abs_shap” column is an array of 1 dimension corresponding to the mean absolute SHAP values across test set subjects (whose SHAP values are concatenated from 5-fold CV) for each fold.

Significant_shap_mean_abs_value_pvalues_1000_random_permut.xlsx contains the mean absolute shap values for each significant ROI (pval <0.05): the columns are : “fold”, and the names of the 17 ROIs rejecting h0 (18 columns total), the rows are : “mean_abs_shap” , “pvalues”, “corrected_pvalues”
