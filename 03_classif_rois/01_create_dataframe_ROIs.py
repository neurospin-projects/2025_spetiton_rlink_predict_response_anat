import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
import os
print(os.getcwd()) # Where am I?
from utils_sara import get_rois


# %%
#inputs
CLINICAL_DATA_DF_FILE = "/neurospin/rlink/participants.tsv"
PATHROI = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_roi/neuromorphometrics_cat12_vbm_roi.tsv"
PATHROI_280ROI = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.7_roi/neuromorphometrics_cat12_vbm_roi.tsv"
PATHROI_LONG ="/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_long_roi/neuromorphometrics_cat12_vbm_roi.tsv"
PATHROI_LONG_280ROI="/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.7_long_roi/neuromorphometrics_cat12_vbm_roi.tsv"
RESPONSE_DATA_DF_FILE = "/neurospin/rlink/REF_DATABASE/phenotype/ecrf/dataset-outcome_version-4.tsv"
QC_FILE = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_qc/qc.tsv"
QC_LONG_FILE = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_long_qc/qc.tsv"

#outputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/03_classif_rois/"
RESULTS_DIR = ROOT+"reports/results_classif/"
DATA_DIR = ROOT+"data/processed/"

"""
difference btw labels at v3 and v4: at m0, one unclassified subject has been classified GR 'sub-80793'
"""
"""
Sara files are here:
/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup/data/processed/df_ROI_age_sex_site_M00_v4labels.csv
"""

# %%
def get_clinical_info(longitudinal, listparticipants):
    # female = 1, male = 0
    df_info_age_sex = pd.read_csv(CLINICAL_DATA_DF_FILE, sep='\t')
    # df_info_age_sex["sex"] = df_info_age_sex["sex"].replace({"M": 0, "F": 1})
    df_info_age_sex = df_info_age_sex.rename(columns={'ses-M00_center': "site_M00"})
    df_info_age_sex = df_info_age_sex.rename(columns={'ses-M03_center': "site_M03"})
    #df_info_age_sex = df_info_age_sex[["participant_id","sex","age","site_M00","site_M03"]]
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

# %%
# DEBUG
verbose=True
longitudinal=False
save=False
WM=False
rois_280 =False
# DEBUG

def save_df_ROI(verbose=True, longitudinal=False, save=False, WM=False, rois_280 =False, notscaled = False):
    """
        longitudinal: (bool) if True, instead of saving ROI at M00 with regular preprocessing, we save ROI of subjects with
                        scans at M00 and M03, where the MRI data was preprocessed longitudinally 
        this function saves all labels as GR, NR, PaR instead of 0, 1 (and if doing multi-class classification, 2 as well)
        WM (bool) : create df of only WM ROI
        rois_280 (bool): if True, create df using cat12 v 12.7. If False, create df using cat12 12.8 measures.
    """

    if longitudinal:
        dfROI = pd.read_csv(PATHROI_LONG,sep='\t') if not rois_280 else pd.read_csv(PATHROI_LONG_280ROI,sep='\t')
        # 190 rows --> 95 subjects with both M00 and M03 data if not rois_280
        # 186 rows --> 93 subjects with both M00 and M03 data if rois_280
        
    else : 
        dfROI = pd.read_csv(PATHROI, sep='\t') if not rois_280 else pd.read_csv(PATHROI_280ROI,sep='\t')
        # 136 rows/subjects of M00 data if not rois_280, otherwise (if rois_280), 135 rows/subjects; PATHROI_V3 should contain the same data 

    print(dfROI)

    all_rois = get_rois(WM) if not rois_280 else [r for r in list(dfROI.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
    all_rois_and_ids = ["participant_id", "tiv"]+all_rois
    
    # dfROI : 415 columns : 136 ROI x 3 + 7 columns (participant_id, session, run, tiv, CSF_Vol, GM_Vol, WM_Vol)
    # 136 ROI x 3 because we have ROIs of GM, CSF, and WM volumes
    # we end up with (136-2)x2 = 268 ROI because we keep GM and CSF volumes only, and the Left vessel and Right vessel ROIs
    # have zero values only

    print("unique sessions : ",dfROI["session"].unique()) #['M00' 'M03']
    # print(dfROI[["Left vessel_WM_Vol","Right vessel_WM_Vol","Left vessel_GM_Vol","Right vessel_GM_Vol","Left vessel_CSF_Vol","Right vessel_CSF_Vol"]])

    if longitudinal :
        participants_M00_and_M03 = [p for p in list(dfROI["participant_id"].values) if set(dfROI[dfROI["participant_id"]==p]["session"].unique())==set(["M00","M03"])]
        print("number of participants with both M00 and M03 data :",len(set(participants_M00_and_M03))) # = 95 subjects for cat 12.8 version, 93 for cat 12.7 version
    else : 
        dfROI = dfROI[dfROI["session"]=="M00"] # technically useless since the file PATHROI should only have M00 data
        if rois_280 : dfROI["participant_id"] = "sub-" + dfROI["participant_id"].astype(str) 

    # 415 columns (longitudinal or not) because we have ['participant_id', 'session', 'run', 'tiv', 'CSF_Vol', 'GM_Vol', 'WM_Vol'] (len=7)
    # and 134 ROI for GM, WM, and CSF volumes => 402 (134*3)
    # and Right vessel and Left vessel (which are in dfROI for GM and WM vand CSF volumes = 6 columns in dfROI) are not in Neuromorphometrics
    # 415 = 134*3 + 7 + 6

   
    if notscaled is False:
    # set tiv value to 1500.0 for all subjects and normalize roi values from all_rois list accordingly
        target_tiv = 1500.0
        scaling_factor = target_tiv / dfROI["tiv"]
        dfROI[all_rois+["tiv"]] = dfROI[all_rois+["tiv"]].mul(scaling_factor, axis=0)
        assert np.allclose(dfROI.tiv, 1500)
    
    # """
    # sex == 0 mean tiv :  1570.6207252230395
    # sex == 1 mean tiv :  1388.4839500725166
    # """

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
        
        if not rois_280: assert set(dfROI["participant_id"])==set(list_participant_ids_qc_1),\
            "the participants of the df of ROI are not the same as those of who passed the quality checks"
        # for rois_280, there are 135 subjects in dfROI but the QC file has 136 because it was run for cat12.8, not cat12.7

    # deal with the response to Li variable / include the labels for classification to the dfROI dataframe
    pop = pd.read_csv(RESPONSE_DATA_DF_FILE, sep='\t')
    
    # pop["participant_id"] = "sub-" + pop["participant_id"].astype(str) # for v3 labels
    pop["participant_id"] = pop["participant_id"].astype(str) # file of v4 labels already has "sub-" substring
    
    # column of population dataframe that defines response to Li label
    label = 'Response.Status.at.end.of.follow.up'

    if verbose:
        # 159 rows in pop df, 131 in dfROI df (counting participants classified "UC")
        dfROI_all_pop = pd.merge(dfROI, pop, on='participant_id', how='inner')
        print("population df columns: ", list(pop.columns))
        print(pop[label].unique())
        print(dfROI_all_pop[label].unique())
        division=2 if longitudinal else 1 # twice the same participant in df for longitudinal df
        print("in population dataframe: \nnumber of Good Responders (GR) :",len(dfROI_all_pop[dfROI_all_pop[label]=="GR"].values)/division)
        print("number of Partial Responders (PaR) :",len(dfROI_all_pop[dfROI_all_pop[label]=="PaR"].values)/division)
        print("number of Non Responders (NR) :",len(dfROI_all_pop[dfROI_all_pop[label]=="NR"].values)/division)
        print("number of UnClassified (UC) :",len(dfROI_all_pop[dfROI_all_pop[label]=="UC"].values)/division)
        
    # we ignore the unclassified subjects, and keep only the good responders, partial responders, and non-responders
    labels_to_keep= ["GR","PaR","NR"]
    pop_to_keep = pop[pop[label].isin(labels_to_keep)]
    pop_to_keep = pop_to_keep[["participant_id",label]]
    # keep the variable 'Response.Status.at.end.of.follow.up' as y (label / outcome) for classification
    pop_to_keep = pop_to_keep.rename(columns={label: "response"})
    assert set(pop_to_keep['response'].unique()) == set(["NR","PaR","GR"])

    print(dfROI) # if longitudinal :  188 rows / 188/2 = 94 subjects ; if not longitudinal : 136 rows/subjects
    df = pd.merge(dfROI, pop_to_keep, on='participant_id', how='inner')
    print(df) # if longitudinal : 182 rows / 182/2 = 91 subjects (after removal of UC) if we also remove PaR, it would be even less subjects (not long: 116 rows/subjects)

    # get clinical data for age, sex, site 
    df_info_age_sex = get_clinical_info(longitudinal, list(dfROI["participant_id"].values))
    df = pd.merge(df, df_info_age_sex[["participant_id","age","sex", "site"]], on='participant_id', how='inner')
    #df['sex'] = df['sex'].replace({0: 'male', 1: 'female'})

    vol_glob = ["tiv", "CSF_Vol", "GM_Vol", "WM_Vol"]

    if longitudinal: df_ROI_age_sex_site = df[all_rois_and_ids+ vol_glob + ["age","sex","site", "session","response"]]
    else : df_ROI_age_sex_site = df[all_rois_and_ids+["age","sex","site","response"]]
    df_ROI_age_sex_site = df_ROI_age_sex_site.reset_index(drop=True)


    if save:
        # save the df
        str_WM = "_WM_Vol" if WM else ""
        filename = f"df_ROI{'-notscaled' if notscaled else ''}_age_sex_site"
        if longitudinal: filename = filename + "_M00_M03"+str_WM+"_v4labels.csv" if not rois_280 else filename + "_M00_M03"+str_WM+"_v4labels_280rois_cat12_7.csv"
        else : filename = filename + "_M00"+str_WM+"_v4labels.csv" if not rois_280 else filename + "_M00"+str_WM+"_v4labels_280rois_cat12_7.csv"
        # df_ROI_age_sex_site['site'] = 'site-' + df_ROI_age_sex_site['site'].astype(str)
        df_ROI_age_sex_site['site'] = df_ROI_age_sex_site['site'].apply(lambda x: f"site-{x:02d}")
        print(df_ROI_age_sex_site)
        df_ROI_age_sex_site.to_csv(DATA_DIR+filename, index=False)
        #df_ROI_age_sex_site.to_csv("new_"+filename, index=False)
        df_ROI_age_sex_site.sex
        print("df saved to : ",DATA_DIR+filename)
        return DATA_DIR + filename



def m3minusm0(WM_roi = False , rois_280 =False):
    """
    Saves a df of M3-M0 ROI measures
        save_m3_minus_m0_df (bool) : if True, save df of differences between m3 and m0 to csv. if False, don't. (no standard scaling at this point) 
        WM_roi (bool) : white matter volumes only
    """

    if WM_roi : path_df_M3minusM0 = DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site_WM_Vol_v4labels.csv" if not rois_280 else DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site_WM_Vol_v4labels_280rois_cat12_7.csv"
    else : path_df_M3minusM0 = DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site_v4labels.csv" if not rois_280 else DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site_v4labels_280rois_cat12_7.csv"

    if not os.path.exists(path_df_M3minusM0):
        if WM_roi: df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03_WM_Vol_v4labels.csv") if not rois_280 else pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03_WM_Vol_v4labels_280rois_cat12_7.csv")
        else : df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03_v4labels.csv") if not rois_280 else pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03_v4labels_280rois_cat12_7.csv")
        dfROIM00 = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M00"].reset_index(drop=True)
        dfROIM03 = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M03"].reset_index(drop=True)

        # Merge M03 and M00 on participant_id
        merged = pd.merge(dfROIM03, dfROIM00, on="participant_id", suffixes=("_M03", "_M00"))

        list_ = ["response","age","sex","site"]
        for l in list_:
            assert (merged[l+'_M00'] == merged[l+'_M03']).all(), " issue with "+l+" between same subjects at M00 and M03"

        columns_M03 = [col for col in merged.columns if col.endswith("_M03")]
        columns_M00 = [col for col in merged.columns if col.endswith("_M00")]
        common_columns = [col[:-4] for col in columns_M03 if col[:-4] + "_M00" in columns_M00]
        print("common_columns ", len(common_columns)) # == 273 since there are 268 ROIs (134 GM and 134 CSF) and y (label), age, sex, site, session
        print(merged)

        # creating a df for differences of M03-M00
        differences_df = pd.DataFrame({
            col: merged[col + "_M03"] - merged[col + "_M00"] for col in common_columns if not col in ["age","sex","site","response","session"]
        })

        differences_df.insert(0, 'participant_id', merged['participant_id'])
        differences_df["response"] = merged["response_M00"]
        differences_df["age"] = merged["age_M00"]
        differences_df["sex"] = merged["sex_M00"]
        differences_df["site"] = merged["site_M00"]

        print("differences_df :",differences_df)
        # df_ROI_age_sex_site['site'] = 'site_' + df_ROI_age_sex_site['site'].astype(str)
        differences_df.to_csv(path_df_M3minusM0,index=False)  

    else : differences_df = pd.read_csv(path_df_M3minusM0)

    if not rois_280: 
        print("differences grouped by response label\n",differences_df[get_rois(WM=WM_roi)+["response"]].groupby("response").median())
        differences_df = differences_df.drop("participant_id", axis=1)
        print(differences_df)

def main():
    # save_df_ROI(WM=False, save=True) 
    # save_df_ROI(WM=False, save=True , longitudinal=True, rois_280=True) 
    # save_df_ROI(WM=True, save=True) 
    # save_df_ROI(WM=True, save=True , longitudinal=True) 

    # %% df_ROI-notscaled_age_sex_site_M00_v4labels.csv
    filename = save_df_ROI(verbose=True, longitudinal=False, save=True, WM=False, rois_280 =False, notscaled=True)
    print(filename)
    """
    /neurospin/signatures/2025_spetiton_rlink_predict_response_anat/03_classif_rois/data/processed/df_ROI-notscaled_age_sex_site_M00_v4labels.csv
    """
    ## Check against Sara
    df = pd.read_csv(filename)
    print(df.groupby('sex')['tiv'].mean())
    """
    sex
    F    1398.766722
    M    1598.805805
    Name: tiv, dtype: float64
    """

    # Check coherence with sara data
    clinic_cols = ["participant_id", 'age', 'sex', 'site', 'response']
    vol_cols = [c for c in df.columns if not (c in clinic_cols)]

    # set tiv value to 1500.0 for all subjects and normalize roi values from all_rois list accordingly
    target_tiv = 1500.0
    scaling_factor = target_tiv / df["tiv"]
    df[vol_cols] = df[vol_cols].mul(scaling_factor, axis=0)
    assert np.allclose(df.tiv, 1500)

    sara = pd.read_csv('/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup/data/processed/df_ROI_age_sex_site_M00_v4labels.csv')
    sara['sex'] = sara['sex'].replace({'male':'M', 'female':'F'})
    vol_glob = ["tiv", "CSF_Vol", "GM_Vol", "WM_Vol"]
    roi_cols = [c for c in sara.columns if not (c in clinic_cols + vol_glob)]
    np.all(sara[clinic_cols] ==  df[clinic_cols])
    np.allclose(sara[roi_cols], df[roi_cols])
    
    # m3minusm0(rois_280 = True)


if __name__ == "__main__":
    main()

# %%
