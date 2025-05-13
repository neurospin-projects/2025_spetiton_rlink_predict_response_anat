import numpy as np, pandas as pd
import xml.etree.ElementTree as ET

from utils import get_rois

#inputs
CLINICAL_DATA_DF_FILE = "/neurospin/rlink/participants.tsv"
PATHROI = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_roi/neuromorphometrics_cat12_vbm_roi.tsv"
PATHROI_LONG ="/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_long_roi/neuromorphometrics_cat12_vbm_roi.tsv"
RESPONSE_DATA_DF_FILE = "/neurospin/rlink/REF_DATABASE/phenotype/ecrf/dataset-outcome_version-3.tsv"
QC_FILE = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_qc/qc.tsv"
QC_LONG_FILE = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_long_qc/qc.tsv"

#outputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
RESULTS_DIR = ROOT+"reports/results_classif/"
DATA_DIR = ROOT+"data/processed/"



def get_clinical_info(longitudinal, listparticipants):
    # female = 1, male = 0
    df_info_age_sex = pd.read_csv(CLINICAL_DATA_DF_FILE, sep='\t')
    df_info_age_sex["sex"] = df_info_age_sex["sex"].replace({"M": 0, "F": 1})
    df_info_age_sex = df_info_age_sex.rename(columns={'ses-M00_center': "site_M00"})
    df_info_age_sex = df_info_age_sex.rename(columns={'ses-M03_center': "site_M03"})
    df_info_age_sex = df_info_age_sex[["participant_id","sex","age","site_M00","site_M03"]]
    df_info_age_sex = df_info_age_sex[df_info_age_sex["participant_id"].isin(listparticipants)]

    # make sure the sites are the same for M00 and M03 (only if longitudinal, as some subjects have been scanned at M00 but not M03)
    if longitudinal: 
        for p in df_info_age_sex["participant_id"].unique():  
            assert (
                df_info_age_sex.loc[df_info_age_sex["participant_id"] == p, "site_M00"].values[0] ==
                df_info_age_sex.loc[df_info_age_sex["participant_id"] == p, "site_M03"].values[0]
            ), f"wrong sites for participant {p}: site_M00 = {df_info_age_sex.loc[df_info_age_sex['participant_id'] == p, 'site_M00'].values[0]}, \
                site_M03 = {df_info_age_sex.loc[df_info_age_sex['participant_id'] == p, 'site_M03'].values[0]}"

        
    df_info_age_sex = df_info_age_sex.rename(columns={'site_M00': "site"})
    df_info_age_sex.drop("site_M03", axis=1, inplace=True)
    df_info_age_sex["site"] = df_info_age_sex["site"].astype(int)

    """
    # just checking the new and old versions contain the same info
    df_info_age_sex_old_version = pd.read_csv(filepath)
    df_info_age_sex_old_version= df_info_age_sex_old_version[["participant_id","site","sex","age"]]
    merged_old_df_new_df = pd.merge(df_info_age_sex, df_info_age_sex_old_version, on ="participant_id",suffixes=("_new", "_old"))
    assert list(merged_old_df_new_df["sex_new"].values) == list(merged_old_df_new_df["sex_old"].values)
    assert list(merged_old_df_new_df["site_new"].values) == list(merged_old_df_new_df["site_old"].values)
    assert list(merged_old_df_new_df["age_new"].values) == list(merged_old_df_new_df["age_old"].values)
    """

    return df_info_age_sex


def save_df_ROI(verbose=True, longitudinal=False, save=False, WM=False):
    """
        longitudinal: (bool) if True, instead of saving ROI at M00 with regular preprocessing, we save ROI of subjects with
                        scans at M00 and M03, where the MRI data was preprocessed longitudinally 
        this function saves all labels as GR, NR, PaR instead of 0, 1 (and if doing multi-class classification, 2 as well)
        WM (bool) : create df of only WM ROI
    """

    all_rois = get_rois(WM)
    all_rois_and_ids = ["participant_id"]+all_rois

    if longitudinal:
        dfROI = pd.read_csv(PATHROI_LONG,sep='\t') # 190 rows --> 95 subjects with both M00 and M03 data
        
    else : 
        dfROI = pd.read_csv(PATHROI, sep='\t') # 136 rows of M00 data --> 136 subjects ; PATHROI_V3 should contain the same data 

    # dfROI : 415 columns : 136 ROI x 3 + 7 columns (participant_id, session, run, tiv, CSF_Vol, GM_Vol, WM_Vol)
    # 136 ROI x 3 because we have ROIs of GM, CSF, and WM volumes
    # we end up with (136-2)x2 = 268 ROI because we keep GM and CSF volumes only, and the Left vessel and Right vessel ROIs
    # have zero values only

    print("unique sessions : ",dfROI["session"].unique())
    # print(dfROI[["Left vessel_WM_Vol","Right vessel_WM_Vol","Left vessel_GM_Vol","Right vessel_GM_Vol","Left vessel_CSF_Vol","Right vessel_CSF_Vol"]])

    if longitudinal :
        participants_M00_and_M03 = [p for p in list(dfROI["participant_id"].values) if set(dfROI[dfROI["participant_id"]==p]["session"].unique())==set(["M00","M03"])]
        print("number of participants with both M00 and M03 data :",len(set(participants_M00_and_M03))) # = 95 subjects
    else : dfROI = dfROI[dfROI["session"]=="M00"] # technically useless since the file PATHROI should only have M00 data

    # 415 columns (longitudinal or not) because we have ['participant_id', 'session', 'run', 'tiv', 'CSF_Vol', 'GM_Vol', 'WM_Vol'] (len=7)
    # and 134 ROI for GM, WM, and CSF volumes => 402 (134*3)
    # and Right vessel and Left vessel (which are in dfROI for GM and WM vand CSF olumes = 6 columns in dfROI) are not in Neuromorphometrics
    # 415 = 134*3 + 7 + 6

    # set tiv value to 1500.0 for all subjects and normalize roi values from all_rois list accordingly
    target_tiv = 1500.0
    scaling_factor = target_tiv / dfROI["tiv"]
    dfROI[all_rois+["tiv"]] = dfROI[all_rois+["tiv"]].mul(scaling_factor, axis=0)

    # make sure the participants passed the quality checks
    if longitudinal: dfQC = pd.read_csv(QC_LONG_FILE, sep='\t')
    else: dfQC = pd.read_csv(QC_FILE, sep='\t')
    
    if longitudinal: assert set(dfQC[dfQC["qc"]==0]["participant_id"].values)==set([41252]),\
        "the subject that did not pass the qc is not the expected one (sub-41252)"
    if not longitudinal: assert dfQC[dfQC["qc"]==0].empty," all subjects have passed the quality check "

    # keep only subjects that passed the quality check
    dfQC["participant_id"] = dfQC["participant_id"].apply(lambda x: "sub-" + str(x))
    dfROI = dfROI[dfROI["participant_id"].isin(dfQC[dfQC["qc"]==1]["participant_id"])]

    if not longitudinal:
        # list_participant_ids_qc_1 = ["sub-"+str(id) for id in list(dfQC[dfQC["qc"]==1]["participant_id"].values)]
        list_participant_ids_qc_1 = [str(id) for id in list(dfQC[dfQC["qc"]==1]["participant_id"].values)]
        
        assert set(dfROI["participant_id"])==set(list_participant_ids_qc_1),\
            "the participants of the df of ROI are not the same as those of who passed the quality checks"

    # deal with the response to Li variable / include the labels for classification to the dfROI dataframe
    pop = pd.read_csv(RESPONSE_DATA_DF_FILE, sep='\t')
    pop["participant_id"] = "sub-" + pop["participant_id"].astype(str)
    
    # column of population dataframe that defines response to Li label
    label = 'Response.Status.at.end.of.follow.up'

    if verbose:
        # 159 rows in pop df, 131 in dfROI df (counting participants classified "UC")
        dfROI_all_pop = pd.merge(dfROI, pop, on='participant_id', how='inner')
        print("population df columns: ", list(pop.columns))
        print(pop[label].unique())
        print(dfROI_all_pop[label].unique())
        print("in population dataframe: \nnumber of Good Responders (GR) :",len(dfROI_all_pop[dfROI_all_pop[label]=="GR"].values))
        print("number of Partial Responders (PaR) :",len(dfROI_all_pop[dfROI_all_pop[label]=="PaR"].values))
        print("number of Non Responders (NR) :",len(dfROI_all_pop[dfROI_all_pop[label]=="NR"].values))
        print("number of UnClassified (UC) :",len(dfROI_all_pop[dfROI_all_pop[label]=="UC"].values))

    # we ignore the unclassified subjects, and keep only the good responders, partial responders, and non-responders
    labels_to_keep= ["GR","PaR","NR"]
    pop_to_keep = pop[pop[label].isin(labels_to_keep)]
    pop_to_keep = pop_to_keep[["participant_id",label]]
    # keep the variable 'Response.Status.at.end.of.follow.up' as y (label / outcome) for classification
    pop_to_keep = pop_to_keep.rename(columns={label: "y"})
    assert set(pop_to_keep['y'].unique()) == set(["NR","PaR","GR"])

    print(dfROI) # if longitudinal :  188 rows / 188/2 = 94 subjects ; if not longitudinal : 136 rows/subjects
    df = pd.merge(dfROI, pop_to_keep, on='participant_id', how='inner')
    print(df) # if longitudinal : 182 rows / 182/2 = 91 subjects (after removal of UC) if we also remove PaR, it would be even less subjects (not long: 116 rows/subjects)

    # get clinical data for age, sex, site 
    df_info_age_sex = get_clinical_info(longitudinal, list(dfROI["participant_id"].values))
    
    df = pd.merge(df, df_info_age_sex[["participant_id","age","sex", "site"]], on='participant_id', how='inner')
    print(df)

    if longitudinal: df_ROI_age_sex_site = df[all_rois_and_ids+["age","sex","site","y", "session"]]
    else : df_ROI_age_sex_site = df[all_rois_and_ids+["age","sex","site","y"]]
    df_ROI_age_sex_site = df_ROI_age_sex_site.reset_index(drop=True)

    print(df_ROI_age_sex_site)
    if save:
        # save the df
        if WM : str_WM = "_WM_Vol"
        if longitudinal: filename= "df_ROI_age_sex_site_M00_M03"+str_WM+".csv"
        else : filename = "df_ROI_age_sex_site_M00"+str_WM+".csv"
        df_ROI_age_sex_site.to_csv(DATA_DIR+filename, index=False)
        print("df saved to : ",filename)


def main():
    save_df_ROI(WM=True, save=True) 
    save_df_ROI(WM=True, save=True , longitudinal=True) 


if __name__ == "__main__":
    main()
