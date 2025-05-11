import os, time, scipy, shap
import numpy as np
import pandas as pd
import nibabel as nib
from utils import read_pkl, get_rois, save_pkl, round_sci
from plots import plot_glassbrain_general

# inputs
ROOT = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
VBMLOOKUP_FILE = "/drf/local/spm12/tpm/labels_Neuromorphometrics.xml"
DATA_DIR = ROOT+"data/processed/"
RESULTS_STATSUNIV_DIR = ROOT+"reports/stats_univ_results/"
FOLDS_DIR = ROOT + "reports/folds_CV/"
# outputs
FEAT_IMPTCE_RES_DIR = ROOT+"reports/feature_importance_results/"

    
def get_mean_abs_shap_array(shap_df):
    """
        returns the mean absolute shap values across all 5-folds CV test sets' shap values for
        dataframe shap_df (in which shap values as arrays for each fold are saved)

        also returns the concatenation of all shap values (sum of # of all test subjects across 5 folds, # roi)
        (116, 268)
    """
    total_nb_subjects = 0
    concat_test_subjects_shap_values = []
    for fold in range(5):
        fold_shap=shap_df[shap_df["fold"]==fold]["res_age_sex_site"].values[0]
        assert fold_shap.shape[1]==268, "there are 268 ROI"
        concat_test_subjects_shap_values.append(fold_shap) # add current fold's test set shap values to the list
        total_nb_subjects+=fold_shap.shape[0]
    assert total_nb_subjects==116
    concat_test_subjects_shap_values= np.concatenate(concat_test_subjects_shap_values,axis=0)
    mean_shap_across_all_folds = np.mean(np.abs(concat_test_subjects_shap_values), axis=0)

    return mean_shap_across_all_folds, concat_test_subjects_shap_values

def create_shap_summary_df_h1_h0(nb_permutations=1000 , cv_folds_seed=1):
    """
        creates and saves a dataframe of two columns "fold" and "mean_abs_shap"
        mean_abs_shap contains mean absolute shap values across all 5-CV folds
        fold is equal to 0 when there is no permutation of classification labels (correct labels) (h1)
        fold != 0 when there is a permuation of classification labels when computing the SHAP values (h0)
        
    nb_permutations (int): nb of permutations 
    cv_folds_seed (int) : 5-fold CV seed used to split data into 5 sets of train/test folds
    """

    df_all_shap = pd.DataFrame(columns=['fold', 'mean_abs_shap']) # df to fill up

    # get shap values computed with correct labels
    shap_file = FEAT_IMPTCE_RES_DIR+"svm_shap_seed"+str(cv_folds_seed)+"_GRvsPaRNR_5fold.pkl"
    shap_df_correct_labels = read_pkl(shap_file)
    mean_shap_across_all_folds, concatenated_shap = get_mean_abs_shap_array(shap_df_correct_labels)

    # save concatenates shap values and corresponding feature values for all test sets of the 5-fold CV 
    # to plot beeswarm shap summary plot
    df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    # get order of participant_ids when concatenating test sets of CV folds in order
    folds_dict = read_pkl(FOLDS_DIR+"subjects_for_each_fold_GRvsPaRNR_5foldCV_seed_1.pkl")
    list_CV_test_participant_ids_in_order = []
    for i in range(5):
        list_CV_test_participant_ids_in_order+=folds_dict[i]["test_subjects_ids"].tolist()

    order_dict = {participant: i for i, participant in enumerate(list_CV_test_participant_ids_in_order)}
    df_sorted = df_ROI_age_sex_site.loc[df_ROI_age_sex_site['participant_id'].map(order_dict).sort_values().index]

    dict_shap_and_features = {"shap values concatenated":concatenated_shap, "features concatenated": df_sorted[get_rois()].values}
    path_dict_shap_and_features= FEAT_IMPTCE_RES_DIR+'dict_shap_and_features_concatenated'+str(nb_permutations)+'_random_permut.pkl'
    # if file doesn't exist in pkl format, also save to pkl
    if not os.path.exists(path_dict_shap_and_features):
        save_pkl(dict_shap_and_features,path_dict_shap_and_features)

    # put it under fold "0" of df_all_shap
    df_all_shap = pd.concat([df_all_shap , pd.DataFrame([{"fold":0, 'mean_abs_shap':mean_shap_across_all_folds}])], axis=0)

    # get shap values computed with nb_permutations permutations of labels
    # save them under folds 1 to nb_permutations+1
    for i in range(1 , nb_permutations+1):
        shap_file_random_permut = FEAT_IMPTCE_RES_DIR+"svm_shap_seed"+str(cv_folds_seed)+"_GRvsPaRNR_5fold_random_permutations_with_seed_"+str(i)+"_of_labels.pkl"
        shap_df_rd_permut = read_pkl(shap_file_random_permut)
        mean_shap_across_all_folds, _ = get_mean_abs_shap_array(shap_df_rd_permut)
        df_all_shap = pd.concat([df_all_shap , pd.DataFrame([{"fold":i, 'mean_abs_shap':mean_shap_across_all_folds}])], axis=0)

    print(df_all_shap)

    # filename to save dataframe summarizing SHAP values 
    path_shap_summary = FEAT_IMPTCE_RES_DIR+'SHAP_summary_res_age_sex_site_h0h1_'+str(nb_permutations)+'_random_permut.xlsx'
    path_shap_summary_pkl = FEAT_IMPTCE_RES_DIR+'SHAP_summary_res_age_sex_site_h0h1_'+str(nb_permutations)+'_random_permut.pkl'

    # if file doesn't exist, save to excel
    if not os.path.exists(path_shap_summary):
        df_all_shap.to_excel(path_shap_summary, index=False)
    # if file doesn't exist in pkl format, also save to pkl
    if not os.path.exists(path_shap_summary_pkl):
        save_pkl(df_all_shap,path_shap_summary_pkl)

def get_pvalues(nb_permutations=1000, alpha=0.05, glassbrain=False):

    # retrieve mean absolute shap values for h1 (fold 0) and h0 (folds 1 to 1001)
    path_shap_summary = FEAT_IMPTCE_RES_DIR+'SHAP_summary_res_age_sex_site_h0h1_'+str(nb_permutations)+'_random_permut.pkl'
    df_all_shap = read_pkl(path_shap_summary)

    roi_names = get_rois()
    assert len(roi_names)==268, "wrong number of ROIs"

    # Unfold the arrays
    shap_expanded_df = pd.DataFrame(
        df_all_shap["mean_abs_shap"].tolist(),  # turn arrays into rows
        columns=roi_names
    )
    shap_expanded_df.insert(0, "fold", df_all_shap["fold"].values) # add fold col back
    # print(shap_expanded_df)

    # Get the h1 row (fold == 0)
    h1_row = shap_expanded_df[shap_expanded_df["fold"] == 0].iloc[0]

    # Compare all rows (h0) to the first row (h1)
    comparison_df = shap_expanded_df[shap_expanded_df["fold"] != 0][roi_names] > h1_row[roi_names]
    # count for each ROI (column) how many rows had higher values than the first row
    # divide by number of rows of comparison_df -1 (nb folds without counting h1 / fold 0)
    pvalues = comparison_df.sum(axis=0) / (len(comparison_df) - 1)

    # append as pvalues a new row
    pvalues_row = pvalues.to_frame().T  # convert to single-row DataFrame
    pvalues_row.insert(0, "fold", "pvalues") # add "pvalues" to "fold" column
    shap_expanded_df = pd.concat([shap_expanded_df, pvalues_row], ignore_index=True)

    shap_expanded_df["row_max"] = shap_expanded_df[roi_names].max(axis=1)
    # last row is "pvalues" and we don't need the max value of the pvalues so we set it to NaN
    shap_expanded_df.loc[shap_expanded_df.index[-1], "row_max"] = np.nan
    # likewise, we don't need the max for h1 (first row), so we set the value to NaN
    shap_expanded_df.loc[shap_expanded_df.index[0], "row_max"] = np.nan


    row_h1_values = shap_expanded_df.iloc[0][roi_names]
    count_max_higher_than_h1 = []
    row_max_values = shap_expanded_df.loc[1:1000, "row_max"].values

    # loop over each ROI column
    for roi in roi_names:
        # count how many "row_max" values are greater than the mean abs shap value for this ROI under h1
        count_higher = (row_max_values > row_h1_values[roi]).sum()
        # divide the count by 1000 and append the result
        count_max_higher_than_h1.append(count_higher / 1000)

    # create a  new row with the computed corrected p-values and add it to df
    # add "corrected_pvalues" for fold column and np.nan for row_max column value
    pvalues_corrected =  ["corrected_pvalues"]+ count_max_higher_than_h1 + [np.nan]  
    pvalues_corrected_df = pd.DataFrame([pvalues_corrected], columns=shap_expanded_df.columns)
    shap_expanded_df = pd.concat([shap_expanded_df, pvalues_corrected_df])
    shap_expanded_df.reset_index(drop=True, inplace=True)

    # select p-values row , and pvalues corrected row
    # Get the row where p-values are stored
    pvalues_row = shap_expanded_df[shap_expanded_df["fold"]=="pvalues"].iloc[0]
    pvalues_corrected_row = shap_expanded_df[shap_expanded_df["fold"]=="corrected_pvalues"].iloc[0]

    # select ROIs with pvalues < alpha (all columns except 'fold')
    uncorrected_significant_roi = [col for col in roi_names if pvalues_row[col] < alpha]
    corrected_significant_roi = [col for col in roi_names if pvalues_corrected_row[col] < alpha]

    print("uncorrected_significant_roi", uncorrected_significant_roi," \nthere are ", len(uncorrected_significant_roi), " uncorrected significant roi.")
    print("\ncorrected_significant_roi", corrected_significant_roi," \nthere are ", len(corrected_significant_roi), " corrected significant roi.\n")
    shap_signficiant_ROI = shap_expanded_df[["fold"]+uncorrected_significant_roi]
    shap_signficiant_ROI = shap_signficiant_ROI[shap_signficiant_ROI["fold"].isin([0,"pvalues","corrected_pvalues"])]
    shap_signficiant_ROI.reset_index(drop=True, inplace=True)

    # order shap_signficiant_ROI in decreasing order
    # mean absolute shap values of h1 order for ROIs (first row of df except for column "fold")
    first_col = shap_signficiant_ROI.columns[0]
    sorted_cols = shap_signficiant_ROI.iloc[0, 1:].sort_values(ascending=False).index.tolist()
    shap_signficiant_ROI = shap_signficiant_ROI[[first_col] + sorted_cols]
    shap_signficiant_ROI[corrected_significant_roi] = shap_signficiant_ROI[corrected_significant_roi].applymap(lambda x: round(x, 4))

    print(shap_signficiant_ROI)

    significant_shap_file = FEAT_IMPTCE_RES_DIR+'significant_shap_mean_abs_value_pvalues_'+str(nb_permutations)+'_random_permut.xlsx'
    # if file doesn't exist, save to excel
    if not os.path.exists(significant_shap_file):
        # atlas_df = pd.read_csv(ROOT+"data/processed/lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=';')
        # roi_names_map = dict(zip(atlas_df['ROI_Neuromorphometrics_labels'], atlas_df['ROIname']))
        # rois = list(shap_signficiant_ROI.columns)
        # rois = [roi for roi in rois if roi!="fold"]
        # roi_names = ["fold"]+[roi_names_map[val] for val in rois]
        # shap_signficiant_ROI.columns = roi_names
        shap_signficiant_ROI.to_excel(significant_shap_file, index=False)
    
    
    mean_abs_significant_uncorrected_shap_values = shap_signficiant_ROI[shap_signficiant_ROI["fold"] == 0].iloc[0]
    roi_dict = {col: mean_abs_significant_uncorrected_shap_values[col] for col in shap_signficiant_ROI.columns if col != "fold"}

    # need to multiply by -1 ROI that appear to have opposite tendencies between shap values and feature value
    roi_dict = {key: roi_dict[key] * -1 if key.endswith("_CSF_Vol") else roi_dict[key] for key in roi_dict}
    for k,v in roi_dict.items():
        print(k,"  ",v)

    if glassbrain :
        blue_roi = ["Left TrIFG triangular part of the inferior frontal gyrus_GM_Vol","Right Hippocampus_GM_Vol",\
                    "Left Hippocampus_GM_Vol","Right AOrG anterior orbital gyrus_GM_Vol","Right Ventral DC_GM_Vol",\
                        "Left Amygdala_GM_Vol", "Right Amygdala_GM_Vol"]
        plot_glassbrain_general(roi_dict, r"uncorrected significant ROI for $\alpha = 0.05$",list_negative=blue_roi)


def plot_beeswarm(nb_permutations=1000, only_significant=False):

    all_rois = get_rois()

    path_dict_shap_and_features= FEAT_IMPTCE_RES_DIR+'dict_shap_and_features_concatenated'+str(nb_permutations)+'_random_permut.pkl'
    dict_shap_and_features = read_pkl(path_dict_shap_and_features)

    shap_arr = dict_shap_and_features["shap values concatenated"]
    features_arr = dict_shap_and_features["features concatenated"]
    csf_indices = [i for i, roi in enumerate(all_rois) if 'CSF_Vol' in roi]
    shap_arrays_negatedCSF = shap_arr.copy()
    shap_arrays_negatedCSF[:, csf_indices] *= -1 # shape (116, 268)

    print(dict_shap_and_features.keys())
    if only_significant: 
        significant_shap_file = FEAT_IMPTCE_RES_DIR+'significant_shap_mean_abs_value_pvalues_'+str(nb_permutations)+'_random_permut.xlsx'
        significant_shap = pd.read_excel(significant_shap_file)
        significant_roi = [roi for roi in list(significant_shap.columns) if roi!="fold"]

        indices = [all_rois.index(item) for item in significant_roi]

        shap_arrays_negatedCSF = shap_arrays_negatedCSF[:,indices]
        features_arr = features_arr[:,indices]

        rois = significant_roi

    else : rois = get_rois()

    
    atlas_df = pd.read_csv(ROOT+"data/processed/lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=';')

    roi_names_map = dict(zip(atlas_df['ROI_Neuromorphometrics_labels'], atlas_df['ROIname']))
    roi_names = [roi_names_map[val] for val in rois]

    shap_arrays_negatedCSF_df = pd.DataFrame(shap_arrays_negatedCSF, columns=roi_names)
    all_folds_Xtest_concatenated_df = pd.DataFrame(features_arr, columns=roi_names)
    print(shap_arrays_negatedCSF_df)
    print(all_folds_Xtest_concatenated_df)

    shap.summary_plot(shap_arrays_negatedCSF_df.values, all_folds_Xtest_concatenated_df.values, \
                        feature_names=shap_arrays_negatedCSF_df.columns)

def main():
    # plot_beeswarm()
    get_pvalues(glassbrain=True)

if __name__ == "__main__":
    main()