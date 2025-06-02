import numpy as np, pandas as pd
from utils import scale_rois_with_tiv, create_folder_if_not_exists
import os, sys
from sklearn.model_selection import train_test_split

# normative modeling with PCNtoolkit
from PCNtoolkit.pcntoolkit.util.utils import create_bspline_basis
from PCNtoolkit.pcntoolkit.normative import estimate
from PCNtoolkit.pcntoolkit.normative import predict

# for residualization on site
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
PATH_TO_DATA_OPENBHB = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/"

# output
DF_FILE = DATA_DIR+"OpenBHB_roi.csv"
DATA_DIR_NM = DATA_DIR+"normative_model/"
NORMATIVE_MODEL_DIR = ROOT + "models/normative_model/"
NORMATIVE_MODEL_RESULTS_DIR = ROOT + "reports/normative_model/"

# no abide2 roi
# 15 datasets


"""
oasis : choose session --> session nb is different for all participants
it seems that the session nb increases as time increases 
(comparison of "session" and "age" columns in participants.csv for the same participant_ids)
we're keeping the earliest (smallest number) session for each participant
regarder pr un participant id ce qui bouge

ICBM : same thing + we keep only "average" (vs "run 1") runs
ADNI : same thing
GSP: same thing
MPI
CoRR
NAR
"""
def merge_to_get_only_first_session(participants_df, roi_data, study_name):
    # Work on a copy to avoid modifying the original too early
    print("study_name ",study_name)
    df = participants_df.copy()
    assert df['session'].apply(lambda x: not isinstance(x, (list, tuple, set))).all(), \
    "Some rows have multiple values in the 'session' column."

    if study_name in ["gsp","mpi","corr","nar"]:
        # keep only session 1 scans
        if study_name=="corr": 
            roi_data = roi_data[(roi_data["session"] == 0) & (roi_data["run"] == 1)]
            roi_df = pd.merge(roi_data , df[["participant_id","age","sex","site","session","run"]], on=["participant_id","session","run"], how="inner")
            roi_df = roi_df.drop(columns='run')

        if study_name=="nar":
            roi_data = roi_data[roi_data["run"] == 1]
            roi_df = pd.merge(roi_data , df[["participant_id","age","sex","site","run"]], on=["participant_id","run"], how="inner")
            roi_df = roi_df.drop(columns='run')

        else : 
            roi_data = roi_data[roi_data["session"]==1]
            roi_df = pd.merge(roi_data , df[["participant_id","age","sex","site","session"]], on=["participant_id","session"], how="inner")
        
        if study_name!="nar": roi_df = roi_df.drop(columns='session')
    
        return roi_df, participants_df


    if study_name=="adni":
        # participant_ids are unique in adni participants.csv file so we can keep the runs from this df only
        # we merge on "run" because both dataframes have unique run values
        
        assert df['run'].apply(lambda x: not isinstance(x, (list, tuple, set))).all(), \
        "ADNI : Some rows have multiple values in the 'run' column."
        assert roi_data['run'].apply(lambda x: not isinstance(x, (list, tuple, set))).all(), \
        "ADNI : Some rows have multiple values in the 'run' column."

        df.rename(columns={'participant_id_x': 'participant_id'}, inplace=True)
        roi_df = pd.merge(roi_data , df[["participant_id","age","sex","site","run"]], on=["participant_id","run"], how="inner")
        roi_df = roi_df.drop(columns='run')
        return roi_df, participants_df

    if study_name=="oasis3":
        # Clean and convert 'session' column
        df['session_clean'] = (
            df['session']
            .str.lstrip('d')
            .str.lstrip('0')
            .replace('', '0')
            .astype(int)
        )

    if study_name=="icbm": 
        df['session_clean'] = (df['session'].astype(int))
        roi_data=roi_data[roi_data["run"]=="average"] # apparently better signal to noise ratio than just one run

    # Get the earliest session per participant_id (even if they appear only once)
    idx = df.groupby('participant_id')['session_clean'].idxmin()

    # Keep only those rows
    participants_df = df.loc[idx].reset_index(drop=True)

    # Drop 'session_clean' if you no longer need it
    participants_df = participants_df.drop(columns='session_clean')

    roi_df = pd.merge(roi_data , participants_df[["participant_id","age","sex","site","session"]], on=["participant_id","session"], how="inner")
    
    # counts = roi_df["session"].value_counts()
    # duplicates = counts[counts > 1]
    # print(duplicates)
    
    roi_df = roi_df.drop(columns='session')

    return roi_df, participants_df

def get_df_one_study_healthy_controls(path_roi, path_participants, study_name):
    # atlas (Neuromorphometrics) dataframe
    atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")
    list_roi_abbr = list(atlas_df["ROIabbr"].values) # roi abbreviations 

    participants_df = pd.read_csv(path_participants)
    participants_df = participants_df[participants_df["diagnosis"]=="control"] # keep only controls
    roi_data = pd.read_csv(path_roi, sep="\t")

    if study_name in ["icbm","oasis3","adni","gsp","mpi","corr","nar"]: # datasets with more than one scan by participant
        roi_df, participants_df = merge_to_get_only_first_session(participants_df, roi_data, study_name)

    if study_name not in ["oasis3","icbm", "adni","gsp","mpi","corr","nar"]: 
        roi_df = pd.merge(roi_data , participants_df[["participant_id","age","sex","site"]], on="participant_id", how="inner")
    
    roi_volumes = [roi for roi in list(roi_df.columns) if roi.endswith("_CSF_Vol") or roi.endswith("_GM_Vol")]
    counts = roi_df["participant_id"].value_counts()
    duplicates = counts[counts > 1]
    
    assert duplicates.empty," some participants have multiple scans data in the dataframe "

    if "tiv" in roi_df.columns: roi_df = roi_df[roi_volumes+["participant_id","age","sex","site","tiv"]]
    if "TIV" in roi_df.columns: roi_df = roi_df[roi_volumes+["participant_id","age","sex","site","TIV"]]
 
    roi_df = scale_rois_with_tiv(roi_df, roi_volumes, target_tiv=1500.0) # scale TIV to 1500.0
    assert len(roi_volumes)==len([roi for roi in roi_volumes if roi in list_roi_abbr]),\
    "the df of oasis database rois doesn't have the same abbreviations as the atlas dataframe"
    assert roi_df["participant_id"].is_unique, "participant_id column must have unique values"
    roi_df["participant_id"] = [f"{study_name}_{i}" for i in roi_df.index]
    roi_df["site"] = study_name+"_" + roi_df["site"].astype(str)

    return roi_df

def save_openBHB_dataframe():

    # paths separate datasets (aggregated to make up OpenBHB)
    biobd_bsnip_roi = pd.read_csv("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/VBMROI_Neuromorphometrics.csv")
    biobd_bsnip_participants = pd.read_csv("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/participantsBD.csv")

    adni_roi = PATH_TO_DATA_OPENBHB+"ADNI_cat12vbm_mwp1_roi.tsv"
    adni_participants = PATH_TO_DATA_OPENBHB+"adni-1st-session_mwp1_participants.csv"

    oasis_participants = PATH_TO_DATA_OPENBHB+"oasis3_t1mri_mwp1_participants.csv"
    oasis_roi = "/neurospin/psy_sbox/oasis3/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"

    icbm_participants = PATH_TO_DATA_OPENBHB+"icbm_t1mri_mwp1_participants.csv"
    icbm_rois = "/neurospin/psy_sbox/icbm/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"

    hcp_participants = PATH_TO_DATA_OPENBHB+"hcp_t1mri_mwp1_participants.csv"
    hcp_roi = "/neurospin/psy_sbox/hcp/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"

    scz_participants = PATH_TO_DATA_OPENBHB+"schizconnect-vip_t1mri_mwp1_participants.csv"
    scz_roi = "/neurospin/psy/schizconnect-vip-prague/derivatives/cat12-12.7_vbm_roi/cat12-12.7_vbm_roi.tsv"

    abide1_roi = "/neurospin/psy_sbox/abide1/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    abide1_participants = PATH_TO_DATA_OPENBHB+"abide1_t1mri_mwp1_participants.csv"

    ixi_roi = "/neurospin/psy_sbox/ixi/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    ixi_participants = PATH_TO_DATA_OPENBHB+"ixi_t1mri_mwp1_participants.csv"

    npc_roi = "/neurospin/psy_sbox/npc/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    npc_participants = PATH_TO_DATA_OPENBHB+"npc_t1mri_mwp1_participants.csv"

    rbp_roi="/neurospin/psy_sbox/rbp/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    rbp_participants = PATH_TO_DATA_OPENBHB+"rbp_t1mri_mwp1_participants.csv"

    gsp_roi="/neurospin/psy_sbox/GSP/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    gsp_participants = PATH_TO_DATA_OPENBHB+"gsp_t1mri_mwp1_participants.csv"

    localizer_roi="/neurospin/psy_sbox/localizer/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    localizer_participants = PATH_TO_DATA_OPENBHB+"localizer_t1mri_mwp1_participants.csv"

    mpi_roi="/neurospin/psy_sbox/mpi-leipzig/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    mpi_participants = PATH_TO_DATA_OPENBHB+"mpi-leipzig_t1mri_mwp1_participants.csv"

    corr_roi="/neurospin/psy_sbox/CoRR/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    corr_participants = PATH_TO_DATA_OPENBHB+"corr_t1mri_mwp1_participants.csv"

    nar_roi="/neurospin/psy_sbox/nar/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
    nar_participants = PATH_TO_DATA_OPENBHB+"nar_t1mri_mwp1_participants.csv"
    if not os.path.exists(DF_FILE):
        print("schizconnect-vip database")
        scz_df = get_df_one_study_healthy_controls(scz_roi, scz_participants, study_name="schizconnect-vip")

        print("hcp database")
        hcp_df = get_df_one_study_healthy_controls(hcp_roi, hcp_participants, study_name="hcp")

        print("OASIS 3 database")
        oasis_df = get_df_one_study_healthy_controls(oasis_roi, oasis_participants, study_name="oasis3")

        print("ICBM database")
        icbm_df = get_df_one_study_healthy_controls(icbm_rois, icbm_participants, study_name="icbm")

        print("BIOBD BSNIP databases")
        atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")
        list_roi_abbr = list(atlas_df["ROIabbr"].values) #roi abbreviations 
        # biobd bsnip databases
        biobd_bsnip_roi_volumes = [roi for roi in list(biobd_bsnip_roi.columns) if roi.endswith("_CSF_Vol") or roi.endswith("_GM_Vol")]
        biobd_bsnip_participants = biobd_bsnip_participants[biobd_bsnip_participants["diagnosis"]=="control"]#  keep only controls
        biobd_bsnip_df = pd.merge(biobd_bsnip_roi, biobd_bsnip_participants[["participant_id","age","sex","site"]], on="participant_id")
        biobd_bsnip_df = biobd_bsnip_df[biobd_bsnip_roi_volumes+["participant_id", "age","sex","site","TIV"]]
        assert len(biobd_bsnip_roi_volumes)==len([roi for roi in biobd_bsnip_roi_volumes if roi in list_roi_abbr]),\
            "the df of oasis database rois doesn't have the same abbreviations as the atlas dataframe"
        assert biobd_bsnip_df["participant_id"].is_unique, "participant_id column must have unique values"
        biobd_bsnip_df["participant_id"] = ["biobd_bsnip_"+str(i) for i in biobd_bsnip_df.index]
        biobd_bsnip_df["site"] = "biobd_bsnip_" + biobd_bsnip_df["site"].astype(str)

        print("ADNI database")
        adni_df = get_df_one_study_healthy_controls(adni_roi, adni_participants, study_name="adni")

        print("ABIDE 1 database")
        abide_df = get_df_one_study_healthy_controls(abide1_roi, abide1_participants, study_name="abide1")

        print("IXI database")
        ixi_df = get_df_one_study_healthy_controls(ixi_roi, ixi_participants, study_name="ixi")

        print("NPC database")
        npc_df = get_df_one_study_healthy_controls(npc_roi, npc_participants, study_name="npc")

        print("rbp database")
        rbp_df = get_df_one_study_healthy_controls(rbp_roi, rbp_participants, study_name="rbp")

        print("gsp database")
        gsp_df = get_df_one_study_healthy_controls(gsp_roi, gsp_participants, study_name="gsp")

        print("localizer database")
        localizer_df = get_df_one_study_healthy_controls(localizer_roi, localizer_participants, study_name="gsp")

        print("MPI database")
        mpi_df = get_df_one_study_healthy_controls(mpi_roi, mpi_participants, study_name="mpi")

        print("CoRR database")
        corr_df = get_df_one_study_healthy_controls(corr_roi, corr_participants, study_name="corr")

        print("NAR database")
        nar_df = get_df_one_study_healthy_controls(nar_roi, nar_participants, study_name="nar")

        list_df = [scz_df, hcp_df, oasis_df, icbm_df, biobd_bsnip_df, adni_df, abide_df,\
                    ixi_df, npc_df, rbp_df, gsp_df, localizer_df, mpi_df, corr_df, nar_df]
        names_databases = ["schizconnect-vip", "hcp", "oasis3", "icbm","biobd_bsnip","adni", "abide","ixi","npc","rbp", "gsp",\
                        "localizer","mpi-leipzig","corr","nar"]
        dict_df = dict(zip(names_databases, list_df))
        print(len(list(dict_df.keys())))
        print(len(names_databases))
        cpt=0
        for df in list_df: 
            assert np.shape(df)[1]==289
            if "TIV" in list(df.columns): df.rename(columns={'TIV': 'tiv'}, inplace=True)
            if cpt==0: list_cols = list(df.columns)
            else: assert list_cols==list(df.columns), "different column names"
            print("\n\ncpt:",cpt, names_databases[cpt])
            cpt+=1

        df_openbhb = pd.concat(list_df, axis=0, ignore_index=True)
        assert df_openbhb[df_openbhb.duplicated(subset=["participant_id", "site"], keep=False)].empty

        # change site names
        unique_sites = df_openbhb["site"].unique()
        # create mapping: site → "openbhb_roi_<index>"
        site_map = {site: f"openbhb_roi_{i}" for i, site in enumerate(unique_sites)}
        df_openbhb["site"] = df_openbhb["site"].map(site_map) # apply mapping
        df_openbhb["site"] = df_openbhb["site"].astype('category').cat.codes

        # convert sex to integer
        df_openbhb["sex"] = df_openbhb["sex"].astype(int)

        # make sure rois are normalized with tiv for everyone
        assert np.isclose(df_openbhb["tiv"], 1500, atol=1e-3).all(), "Not all 'tiv' values are close to 1500"

        # assign unique integer to each participant_id
        df_openbhb["participant_id"] = df_openbhb["participant_id"].astype("category").cat.codes

        # move participant_id to the first column
        cols = ["participant_id"] + [col for col in df_openbhb.columns if col != "participant_id"]
        df_openbhb = df_openbhb[cols]

        # remove columns with only zero values
        zero_columns = [col for col in df_openbhb.columns if (df_openbhb[col] == 0).all()] #['lInfLatVen_GM_Vol', 'lOC_GM_Vol', 'lInfLatVen_CSF_Vol', 'lOC_CSF_Vol']
        df_openbhb = df_openbhb.drop(columns=zero_columns)

        duplicates = df_openbhb.columns[df_openbhb.columns.duplicated()].tolist()
        assert duplicates==[],"there are duplicates in the df"

        # save df to csv    
        df_openbhb.to_csv(DF_FILE, index=False) 
        print(df_openbhb)

def residualize_four_regions_df(df_selected_regions, selected_roi_names, \
    formula_res="site", formula_full = "age + sex + site", no_res=False):
    """
    returns a dataframe of as many columns as there are selected_roi_names,
    with roi residualized on site 
    """
    if no_res: 
        features = df_selected_regions[selected_roi_names].to_numpy()
        df_features_residualized = pd.DataFrame(features, columns=selected_roi_names)
        return df_features_residualized
    residualizer = Residualizer(data=df_selected_regions, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(df_selected_regions)
    features = df_selected_regions[selected_roi_names].to_numpy()
    residualizer.fit(features, Zres)
    features_res = residualizer.transform(features, Zres)
    df_features_residualized = pd.DataFrame(features_res, columns=selected_roi_names)
    return df_features_residualized


def hippo_amyg_save_for_normative_model(no_res=True):
    """
    select the four rois we are interested in for normative modeling:
    gray matter volume of bilateral hippocampus and bilateral amygdala

    residualize on site variable with formula y ~ intercept + age + sex + site
    
    """
    assert os.path.exists(DF_FILE),"openBHB roi dataframe file does not exist"
    df_openbhb = pd.read_csv(DF_FILE)
    print("reading ",DF_FILE," ...")
    print("dataframe OpenBHB rois ...\n",df_openbhb)
    quit()

    # select the regions we're interested in (bilateral hippocampus and amygdala gray matter volumes)
    four_regions_of_interest = ["lHip_GM_Vol","rHip_GM_Vol","lAmy_GM_Vol","rAmy_GM_Vol"]#, "lPal_GM_Vol","rPal_GM_Vol",\
        #'lPut_GM_Vol', 'rPut_GM_Vol']
    # rename OpenBHB names for these roi into neuromorphometrics names used for RLink dataframe 
    print("\nHippocampus and Amygdala regions only : \n", df_openbhb)
    atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")
    atlas_df_four_roi = atlas_df[atlas_df["ROIabbr"].isin(four_regions_of_interest)]
    dict_change_names = dict(zip(atlas_df_four_roi["ROIabbr"],atlas_df_four_roi["ROI_Neuromorphometrics_labels"]))

    df_openbhb_four_regions  = df_openbhb[four_regions_of_interest+["age","sex","site"]]
    df_openbhb_four_regions = df_openbhb_four_regions.rename(columns=dict_change_names)
    print("\nSetting names to the same neuromorphometrics atlas names as in the Rlink dataframes:\n",df_openbhb_four_regions)

    # Bin age (into decades) (so that there are no classes with one subject only for age + sex)
    df_openbhb_four_regions['age_bin'] = pd.cut(df_openbhb_four_regions['age'], bins=[0, 20, 30, 40, 50, 60, 70, 100], labels=False)

    # Create stratify label on binned age + sex
    df_openbhb_four_regions['stratify_label'] = df_openbhb_four_regions['age_bin'].astype(str) + "_" + df_openbhb_four_regions['sex'].astype(str)

    # Train-test split stratified on age and sex
    df_openbhb_tr, df_openbhb_te = train_test_split(
        df_openbhb_four_regions,
        test_size=0.2,
        stratify=df_openbhb_four_regions['stratify_label'],
        random_state=42  # to ensure reproducibility
    )

    df_openbhb_tr.drop(columns=['stratify_label', 'age_bin'], inplace=True)
    df_openbhb_te.drop(columns=['stratify_label', 'age_bin'], inplace=True)

    # print(df_openbhb_tr["age"].mean(), df_openbhb_te["age"].mean()) # 33.88 mean age, 33.95 mean age
    # print((df_openbhb_tr['sex'] == 1).mean(),(df_openbhb_te['sex'] == 1).mean()) # 51.77% women, 51.76% women

    four_roi_names = dict_change_names.values()
    # openBHB : residualize roi on site
    openBHB_features_residualized_tr = residualize_four_regions_df(df_openbhb_tr, four_roi_names,no_res=no_res)
    openBHB_features_residualized_te = residualize_four_regions_df(df_openbhb_te, four_roi_names,no_res=no_res)

    # print("\nOpenBHB tr roi now residualized on site ...\n",openBHB_features_residualized_tr)
    # print("\nOpenBHB te roi now residualized on site ...\n",openBHB_features_residualized_te)

    # openBHB : covariates : age and sex
    openBHB_covariates_tr = df_openbhb_tr[["age","sex"]]
    openBHB_covariates_te = df_openbhb_te[["age","sex"]]
    # print("\nseparate from their covariates...\n",openBHB_covariates_tr)

    print("OpenBHB train features (residualized on site) shape ",np.shape(openBHB_features_residualized_tr), \
          "\n,and covariates ",np.shape(openBHB_covariates_tr))
    print("OpenBHB test features (residualized on site) shape ",np.shape(openBHB_features_residualized_te), \
          "\n,and covariates ",np.shape(openBHB_covariates_te))
    
    # read Rlink dataframe at M0
    df_ROI_age_sex_site_rlink_M0 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    # print(df_ROI_age_sex_site_rlink_M0["participant_id"])
    df_ROI_age_sex_site_rlink_four_regions_M0 = df_ROI_age_sex_site_rlink_M0[four_roi_names].copy()
    df_ROI_age_sex_site_rlink_four_regions_M0[["age","sex","site"]] = df_ROI_age_sex_site_rlink_M0[["age","sex","site"]]
    print("\nRLink M0 roi with age, sex, site ...\n",df_ROI_age_sex_site_rlink_four_regions_M0)

    # rlink M0: residualize roi on site
    rlink_features_residualized_M0 = residualize_four_regions_df(df_ROI_age_sex_site_rlink_four_regions_M0, four_roi_names,no_res=no_res)
    print("\nRlink M0 roi now residualized on site ...\n",rlink_features_residualized_M0)
    # rlink M0: covariates : age and sex
    rlink_covariates_M0 = df_ROI_age_sex_site_rlink_four_regions_M0[["age","sex"]]
    print("RLink features (residualized on site) shape ",np.shape(rlink_features_residualized_M0), \
          "\n,and covariates ",np.shape(rlink_covariates_M0))
    
    df_ROI_age_sex_site_rlink_M3 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03.csv")
    df_ROI_age_sex_site_rlink_M3 = df_ROI_age_sex_site_rlink_M3[df_ROI_age_sex_site_rlink_M3["session"]=="M03"]
    # print(df_ROI_age_sex_site_rlink_M3["participant_id"])
    df_ROI_age_sex_site_rlink_four_regions_M3 = df_ROI_age_sex_site_rlink_M3[four_roi_names].copy()
    df_ROI_age_sex_site_rlink_four_regions_M3[["age","sex","site"]] = df_ROI_age_sex_site_rlink_M3[["age","sex","site"]]
    print("\nRLink M3 roi with age, sex, site ...\n",df_ROI_age_sex_site_rlink_four_regions_M3)

    # rlink M3: residualize roi on site
    rlink_features_residualized_M3 = residualize_four_regions_df(df_ROI_age_sex_site_rlink_four_regions_M3, four_roi_names,no_res=no_res)
    print("\nRlink M0 roi now residualized on site ...\n",rlink_features_residualized_M3)
    # rlink M3: covariates : age and sex
    rlink_covariates_M3 = df_ROI_age_sex_site_rlink_four_regions_M3[["age","sex"]]
    print("RLink features (residualized on site) shape ",np.shape(rlink_features_residualized_M3), \
          "\n,and covariates ",np.shape(rlink_covariates_M3))

    no_res_str ="_no_res" if no_res else ""

    openBHB_cov_tr, openBHB_feat_tr = DATA_DIR_NM   + 'cov_openBHB_tr.txt', DATA_DIR_NM + 'resp_openBHB_tr'+no_res_str+'.txt'
    openBHB_cov_te, openBHB_feat_te = DATA_DIR_NM   + 'cov_openBHB_te.txt', DATA_DIR_NM + 'resp_openBHB_te'+no_res_str+'.txt'

    if not os.path.exists(openBHB_cov_tr): openBHB_covariates_tr.to_csv(openBHB_cov_tr, sep = '\t', header=False, index=False)
    if not os.path.exists(openBHB_feat_tr): openBHB_features_residualized_tr.to_csv(openBHB_feat_tr, sep = '\t', header=False, index=False)

    if not os.path.exists(openBHB_cov_te): openBHB_covariates_te.to_csv(openBHB_cov_te, sep = '\t', header=False, index=False)
    if not os.path.exists(openBHB_feat_te): openBHB_features_residualized_te.to_csv(openBHB_feat_te, sep = '\t', header=False, index=False)

    rlinkM0_cov, rlinkM0_feat = DATA_DIR_NM   + 'cov_rlinkM0.txt', DATA_DIR_NM + 'resp_rlinkM0'+no_res_str+'.txt'
    if not os.path.exists(rlinkM0_cov): rlink_covariates_M0.to_csv(rlinkM0_cov, sep = '\t', header=False, index=False)
    if not os.path.exists(rlinkM0_feat): rlink_features_residualized_M0.to_csv(rlinkM0_feat, sep = '\t', header=False, index=False)

    rlinkM3_cov, rlinkM3_feat = DATA_DIR_NM   + 'cov_rlinkM3.txt', DATA_DIR_NM + 'resp_rlinkM3'+no_res_str+'.txt'
    if not os.path.exists(rlinkM3_cov): rlink_covariates_M3.to_csv(rlinkM3_cov, sep = '\t', header=False, index=False)
    if not os.path.exists(rlinkM3_feat): rlink_features_residualized_M3.to_csv(rlinkM3_feat, sep = '\t', header=False, index=False)


def create_bspline_basis_for_covariates(rlink_age_range=True, openBHB_age_range=False):
    """
    create bspline bases for covariates
    openBHB age range is default for bspline bases
    choose either age range of openBHB or rlink for bsplines min and max for age covariate
    
    """
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "

    openBHB_cov_tr = DATA_DIR_NM   + 'cov_openBHB_tr.txt'
    openBHB_cov_te = DATA_DIR_NM   + 'cov_openBHB_te.txt'
    rlinkM0_cov = DATA_DIR_NM   + 'cov_rlinkM0.txt'
    rlinkM3_cov = DATA_DIR_NM   + 'cov_rlinkM3.txt'

    # load train covariate data matrices
    openBHB_covariates_tr = np.loadtxt(os.path.join(openBHB_cov_tr))
    openBHB_covariates_te = np.loadtxt(os.path.join(openBHB_cov_te))
    rlink_covariates_M0 = np.loadtxt(os.path.join(rlinkM0_cov))
    rlink_covariates_M3 = np.loadtxt(os.path.join(rlinkM3_cov))

    # add intercept column
    openBHB_covariates_tr = np.concatenate((openBHB_covariates_tr, np.ones((openBHB_covariates_tr.shape[0],1))), axis=1)
    openBHB_covariates_te = np.concatenate((openBHB_covariates_te, np.ones((openBHB_covariates_te.shape[0],1))), axis=1)

    rlink_covariates_M0 = np.concatenate((rlink_covariates_M0, np.ones((rlink_covariates_M0.shape[0],1))), axis=1)
    rlink_covariates_M3 = np.concatenate((rlink_covariates_M3, np.ones((rlink_covariates_M3.shape[0],1))), axis=1)
    
    df_openbhb = pd.read_csv(DF_FILE)

    age_min = round(min(list(df_openbhb["age"].unique())),1)
    age_max = round(max(list(df_openbhb["age"].unique())),1)
    print("OpenBHB min and max age :", age_min, age_max) # 6.5 95.0
    if openBHB_age_range: B = create_bspline_basis(age_min, age_max)

    df_ROI_age_sex_site_rlink_M0 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    age_min = round(min(list(df_ROI_age_sex_site_rlink_M0["age"].unique())),1)
    age_max = round(max(list(df_ROI_age_sex_site_rlink_M0["age"].unique())),1)
    print("RLink min and max age :", age_min, age_max) # 18 69
    if rlink_age_range: B = create_bspline_basis(age_min, age_max)

    # create the basis expansion for the covariates
    print('Creating basis expansion ...')

    # create Bspline basis set
    Phi_openBHB_tr = np.array([B(i) for i in openBHB_covariates_tr[:,0]]) # age is the first columns
    Phi_openBHB_te = np.array([B(i) for i in openBHB_covariates_te[:,0]])
    Phi_rlink_M0 = np.array([B(i) for i in rlink_covariates_M0[:,0]])
    Phi_rlink_M3 = np.array([B(i) for i in rlink_covariates_M3[:,0]])

    openBHB_covariates_tr = np.concatenate((openBHB_covariates_tr, Phi_openBHB_tr), axis=1)
    openBHB_covariates_te = np.concatenate((openBHB_covariates_te, Phi_openBHB_te), axis=1)
    rlink_covariates_M0 = np.concatenate((rlink_covariates_M0, Phi_rlink_M0), axis=1)
    rlink_covariates_M3 = np.concatenate((rlink_covariates_M3, Phi_rlink_M3), axis=1)

    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"

    openBHB_cov_tr = DATA_DIR_NM   + 'cov_openBHB_tr_bspline'+str_age_range+'.txt'
    openBHB_cov_te = DATA_DIR_NM   + 'cov_openBHB_te_bspline'+str_age_range+'.txt'
    rlinkM0_cov = DATA_DIR_NM   + 'cov_rlinkM0_bspline'+str_age_range+'.txt'
    rlinkM3_cov = DATA_DIR_NM   + 'cov_rlinkM3_bspline'+str_age_range+'.txt'

    if not os.path.exists(openBHB_cov_tr): np.savetxt(openBHB_cov_tr, openBHB_covariates_tr)
    if not os.path.exists(openBHB_cov_te): np.savetxt(openBHB_cov_te, openBHB_covariates_te)
    if not os.path.exists(rlinkM0_cov): np.savetxt(rlinkM0_cov, rlink_covariates_M0)
    if not os.path.exists(rlinkM3_cov): np.savetxt(rlinkM3_cov, rlink_covariates_M3)

    print("...done.")
    
def train_normative_model_wblr(rlink_age_range=False, openBHB_age_range=True, no_bspline=False, no_res=True):
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "
    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"
    if no_bspline: str_age_range="_no_bspline"
    no_res_str ="_no_res" if no_res else ""

    openBHB_feat_tr = DATA_DIR_NM + 'resp_openBHB_tr'+no_res_str+'.txt'
    openBHB_feat_te = DATA_DIR_NM + 'resp_openBHB_te'+no_res_str+'.txt'

    if no_bspline: openBHB_cov_tr, openBHB_cov_te = DATA_DIR_NM + 'cov_openBHB_tr.txt', DATA_DIR_NM + 'cov_openBHB_te.txt'
    else : 
        openBHB_cov_tr = DATA_DIR_NM   + 'cov_openBHB_tr_bspline'+str_age_range+'.txt'
        openBHB_cov_te = DATA_DIR_NM   + 'cov_openBHB_te_bspline'+str_age_range+'.txt'

    covtr = np.loadtxt(openBHB_cov_tr,dtype=str)
    covte = np.loadtxt(openBHB_cov_te,dtype=str)

    print("covariates :",np.shape(covtr), np.shape(covte))
    print("covariates :",type(covtr), type(covte))

    # load train & test response files
    resptr = np.loadtxt(openBHB_feat_tr,dtype=float)
    respte = np.loadtxt(openBHB_feat_te,dtype=float)
    print("responses :",np.shape(resptr), np.shape(respte), type(resptr), type(respte))
    modelname = "wblr_bilateral_hippo_amyg_openBHB_roi"+str_age_range+no_res_str
    path_model = NORMATIVE_MODEL_DIR+modelname
    create_folder_if_not_exists(path_model)
    os.chdir(path_model)
    estimate(openBHB_cov_tr,
            openBHB_feat_tr,
            testresp=openBHB_feat_te,
            testcov=openBHB_cov_te,
            alg = "blr",
            optimizer = 'powell',
            savemodel=True,
            saveoutput=True,
            standardize=False, warp = "WarpSinArcsinh")


def evaluate_rlink_four_rois_using_normative_model(rlink_age_range=False, openBHB_age_range=True, no_bspline=False, no_res=True):
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "
    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"
    if no_bspline: str_age_range="_no_bspline"
    no_res_str ="_no_res" if no_res else ""

    rlinkM0_feat, rlinkM3_feat = DATA_DIR_NM + 'resp_rlinkM0'+no_res_str+'.txt', DATA_DIR_NM + 'resp_rlinkM3'+no_res_str+'.txt'

    if no_bspline: rlinkM0_cov, rlinkM3_cov = DATA_DIR_NM + 'cov_rlinkM0.txt', DATA_DIR_NM + 'cov_rlinkM3.txt'
    else: 
        rlinkM0_cov = DATA_DIR_NM + 'cov_rlinkM0_bspline'+str_age_range+'.txt'
        rlinkM3_cov = DATA_DIR_NM + 'cov_rlinkM3_bspline'+str_age_range+'.txt'

    M0cov = np.loadtxt(rlinkM0_cov,dtype=str)
    M3cov = np.loadtxt(rlinkM3_cov,dtype=str)

    M0feat = np.loadtxt(rlinkM0_feat,dtype=str)
    M3feat = np.loadtxt(rlinkM3_feat,dtype=str)

    print("covariates M0 :",np.shape(M0cov), np.shape(M0cov))
    print("covariates M3 :",type(M3cov), type(M3cov))
    print("responses/features M0 :",np.shape(M0feat), np.shape(M0feat))
    print("responses/features M3 :",type(M3feat), type(M3feat))

    modelname = "wblr_bilateral_hippo_amyg_openBHB_roi"+str_age_range
    path_model = NORMATIVE_MODEL_DIR+modelname
    path_results_M0, path_results_M3 = NORMATIVE_MODEL_RESULTS_DIR+"M0"+str_age_range+no_res_str, NORMATIVE_MODEL_RESULTS_DIR+"M3"+str_age_range+no_res_str
    create_folder_if_not_exists(path_results_M0)
    create_folder_if_not_exists(path_results_M3)

    if not os.listdir(path_results_M0): # if folder is empty
        predict(rlinkM0_cov,
                alg='blr',
                respfile=rlinkM0_feat,
                model_path= path_model+"/Models",
                save_path = path_results_M0)
        
    if not os.listdir(path_results_M3): # if folder is empty
        predict(rlinkM3_cov,
            alg='blr',
            respfile=rlinkM3_feat,
            model_path= path_model+"/Models",
            save_path = path_results_M3)

def read_results(label=1, rlink_age_range=False, openBHB_age_range=True, no_bspline=False, no_res=True):
    no_res_str ="_no_res" if no_res else ""
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "
    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"
    if no_bspline: str_age_range="_no_bspline"

    z_scores_M0 = np.loadtxt(NORMATIVE_MODEL_RESULTS_DIR+"M0"+str_age_range+no_res_str+"/Z_predict.txt",dtype=str)
    z_scores_M3 = np.loadtxt(NORMATIVE_MODEL_RESULTS_DIR+"M3"+str_age_range+no_res_str+"/Z_predict.txt",dtype=str)
    # print(np.shape(z_scores_M0))
    # print(np.shape(z_scores_M3))
    atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")
    four_regions_of_interest = ["lHip_GM_Vol","rHip_GM_Vol","lAmy_GM_Vol","rAmy_GM_Vol"]  #,"lPal_GM_Vol","rPal_GM_Vol",\
       # 'lPut_GM_Vol', 'rPut_GM_Vol']
    atlas_df_four_roi = atlas_df[atlas_df["ROIabbr"].isin(four_regions_of_interest)]
    dict_change_names = dict(zip(atlas_df_four_roi["ROIabbr"],atlas_df_four_roi["ROI_Neuromorphometrics_labels"]))
    four_roi_names = dict_change_names.values()
    df_zscores_M0 = pd.DataFrame(z_scores_M0, columns=four_roi_names)
    df_zscores_M3 = pd.DataFrame(z_scores_M3, columns=four_roi_names)

    # print(df_zscores_M0)
    # print(df_zscores_M3)

    print(df_zscores_M0.median(), "\n",df_zscores_M3.median(),"\n")
    
    df_ROI_age_sex_site_rlink_M0 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")

    df_ROI_age_sex_site_rlink_M3 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03.csv")
    df_ROI_age_sex_site_rlink_M3 = df_ROI_age_sex_site_rlink_M3[df_ROI_age_sex_site_rlink_M3["session"]=="M03"]
    df_ROI_age_sex_site_rlink_M3.reset_index(inplace=True)
    df_ROI_age_sex_site_rlink_M3["y"] = df_ROI_age_sex_site_rlink_M3["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    participant_ids_label = df_ROI_age_sex_site_rlink_M3.loc[df_ROI_age_sex_site_rlink_M3['y'] == label, 'participant_id'].tolist()

    df_zscores_M0["participant_id"]=df_ROI_age_sex_site_rlink_M0["participant_id"].copy()
    df_zscores_M3["participant_id"]=df_ROI_age_sex_site_rlink_M3["participant_id"].copy()

    # add participant_id to zscores dataframes
    merged = pd.merge(df_zscores_M0, df_zscores_M3, on='participant_id', suffixes=('_M0', '_M3'))
    merged = merged[merged["participant_id"].isin(participant_ids_label)]
    merged.reset_index(inplace=True)

    closer_to_zero = pd.DataFrame()
    # print(merged)

    closer_to_zero['participant_id'] = merged['participant_id']

    for col in df_zscores_M0.columns:
        if col != 'participant_id':
            col_M0 = pd.to_numeric(merged[f'{col}_M0'], errors='coerce')
            col_M3 = pd.to_numeric(merged[f'{col}_M3'], errors='coerce')
            
            # Compare absolute values: which is closer to zero?
            closer = np.where(np.abs(col_M0) < np.abs(col_M3), 'M0', 'M3')
            
            closer_to_zero[col] = closer

    counts = closer_to_zero.drop(columns='participant_id').apply(pd.Series.value_counts)
    print(counts.T)  
    n_total = len(closer_to_zero)

    # Count of M3 for each column (i.e., number of times M3 is closer to 0)
    m3_counts = (closer_to_zero.drop(columns='participant_id') == 'M3').sum()

    # Convert to percentage
    m3_percentages = (m3_counts / n_total) * 100

    # Display result
    print(m3_percentages.round(2).to_frame(name='% M3 closer to 0'))

    for col in closer_to_zero.columns:
        if col != 'participant_id':
            if label ==1:
                print(f'\nParticipants where M0 is closer to 0 than M3 for region: {col}')
                ids = closer_to_zero.loc[closer_to_zero[col] == 'M0', 'participant_id']
                print(ids.tolist())
            if label == 0:
                print(f'\nParticipants where M3 is closer to 0 than M0 for region: {col}')
                ids = closer_to_zero.loc[closer_to_zero[col] == 'M3', 'participant_id']
                print(ids.tolist())
                tot = len(df_ROI_age_sex_site_rlink_M0[df_ROI_age_sex_site_rlink_M0["participant_id"].isin(ids.tolist())]["y"])
                n_par = df_ROI_age_sex_site_rlink_M0[
                    df_ROI_age_sex_site_rlink_M0["participant_id"].isin(ids.tolist())
                ]["y"].eq("PaR").sum()
                n_nr = df_ROI_age_sex_site_rlink_M0[
                    df_ROI_age_sex_site_rlink_M0["participant_id"].isin(ids.tolist())
                ]["y"].eq("NR").sum()
                print("percentage of NR ",round(100*(n_nr/tot),2))
                print("percentage of PaR ",round(100*(n_par/tot),2))



def main():
    # save_openBHB_dataframe() # to save OpenBHB roi dataframe if it doesn't already exist
    # hippo_amyg_save_for_normative_model()
    # create_bspline_basis_for_covariates()
    # train_normative_model_wblr()
    # evaluate_rlink_four_rois_using_normative_model()
    read_results(label = 0)
    quit()
    """
    check participant_id order for NM and read results
    redo eval without site residualization
    try without bspline basis
    , or with ages from rlink instead of openbhb
    try without warping
    try with scikit learn
    """

if __name__ == "__main__":
    main()




