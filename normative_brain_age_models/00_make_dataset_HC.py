import numpy as np, pandas as pd
import os
import matplotlib.pyplot as plt
from utils import scale_rois_with_tiv, get_lists_roi_in_both_openBHB_and_rlink

# inputs OpenBHB df
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
PATH_TO_DATA_OPENBHB = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/"

# inputs OpenBHB aggregated with other cohorts df
PARTICIPANTS_FILE_OPENMIND = "/neurospin/psy/openmind/participants.tsv"
ROI_FILE_OPENMIND = "/neurospin/psy/openmind/derivatives/cat12-12.8.2_vbm_roi/neuromorphometrics_cat12_vbm_roi.tsv"
ROI_HCP_FILE = "/neurospin/psy/hcp/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
PARTICIPANTS_HCP = "/neurospin/psy/hcp/HCP_t1mri_mwp1_participants_v-2023.csv" #/neurospin/psy/hcp/participants.tsv"
OPENBHB_DATAFRAME = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/data/processed/OpenBHB_roi.csv"

# output creation OpenBHB df
DF_FILE = DATA_DIR+"OpenBHB_roi.csv"

# output creation OpenBHB aggregated with other cohorts df
DF_PATH = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/data/processed/hcp_open_mind_roi_vbm_with_age.csv"
DF_PATH_WITH_OPENBHB = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/data/processed/hcp_open_mind_open_bhb_roi_vbm_with_age.csv"


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

def plot_age_distribution(df, cohort_name):
    """
    Plot the age distribution of the 'age' column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing an 'age' column.
    """
    if "age" not in df.columns:
        raise ValueError("The DataFrame must contain an 'age' column.")

    plt.figure(figsize=(8, 5))
    plt.hist(df["age"].dropna(), bins=30, edgecolor="black", alpha=0.7)
    plt.title(cohort_name+" age distribution", fontsize=14)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

def create_df(save=False):
    # ============== Open Mind ======================


    # 1. select healthy participants
    participants_open_mind = pd.read_csv(PARTICIPANTS_FILE_OPENMIND, sep="\t")
    # print(participants["health_status"].unique()) #[nan 'healthy' 'ill']
    participants_open_mind = participants_open_mind[participants_open_mind["health_status"]=="healthy"]
    # select subjects with age available
    participants_open_mind = participants_open_mind[participants_open_mind["age"].notna()]
    # plot_age_distribution(participants_open_mind, "open mind")
    rois_open_mind = pd.read_csv(ROI_FILE_OPENMIND, sep="\t")
    rois_open_mind["participant_id"] = rois_open_mind["dataset"].astype(str) + "__" + rois_open_mind["participant_id"].astype(str)
    # list of rois (GM and CSF volumes)
    all_rois_open_mind = [ r for r in list(rois_open_mind.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]

    # 2. scale by TIV such that TIV is equal to 1500.0
    target_tiv = 1500.0
    scaling_factor = target_tiv / rois_open_mind["tiv"]
    rois_open_mind[all_rois_open_mind+["tiv"]] = rois_open_mind[all_rois_open_mind+["tiv"]].mul(scaling_factor, axis=0)

    # 3. add "age" column to df
    participants_open_mind["sex"] = participants_open_mind["sex"].map({"female": 1, "male": 0})
    rois_open_mind = pd.merge(rois_open_mind, participants_open_mind[["participant_id","age","sex"]], on="participant_id", how="inner")
    list_roi_openbhb, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    # print(len(list_roi_rlink), len(list_roi_openbhb)) # 252 for both

    # 4. rename rois to match openbhb roi names
    dict_rlink_to_openbhb_roi_names = dict(zip(list_roi_rlink, list_roi_openbhb)) # get dict of roi names correspondencies
    rois_open_mind = rois_open_mind[list_roi_rlink+["age","sex"]]
    rois_open_mind = rois_open_mind.rename(columns=dict_rlink_to_openbhb_roi_names)

    # 5. add cohort column
    rois_open_mind["cohort"]="open_mind"

    # ============== HCP ======================

    rois_hcp = pd.read_csv(ROI_HCP_FILE, sep="\t")
    # get all roi names in hcp VBM roi dataframe
    all_rois_hcp = [ r for r in list(rois_hcp.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
    # 1. select healthy participants
    participants_hcp = pd.read_csv(PARTICIPANTS_HCP, sep="\t")
    participants_hcp = participants_hcp[participants_hcp["diagnosis"]=="control"]
    
    rois_hcp = pd.merge(rois_hcp, participants_hcp[["participant_id","age","sex"]], on="participant_id", how="inner")

    # 2. scale by TIV such that TIV is equal to 1500.0
    target_tiv = 1500.0
    scaling_factor = target_tiv / rois_hcp["TIV"]
    rois_hcp[all_rois_hcp+["TIV"]] = rois_hcp[all_rois_hcp+["TIV"]].mul(scaling_factor, axis=0)

    # 3. add "age" column to df
    # plot_age_distribution(rois_hcp, "hcp")
    rois_hcp = rois_hcp[list_roi_openbhb+["age","sex"]]

    # 4. add cohort column
    rois_hcp["cohort"]="hcp"

    # concatenate
    df_all = pd.concat([rois_hcp, rois_open_mind], axis=0).reset_index(drop=True)

    # plot_age_distribution(df_all, "open mind + hcp")
    df_all.to_csv(DF_PATH, index=False)
    df_openbhb = pd.read_csv(OPENBHB_DATAFRAME)
    df_openbhb = df_openbhb[list_roi_openbhb+["age","sex","site"]]
    df_all = df_all.rename(columns={"cohort": "site"})
    print(df_all)
    df_all_with_openbhb = pd.concat([df_all, df_openbhb], axis=0).reset_index(drop=True)
    df_all_with_openbhb.to_csv(DF_PATH_WITH_OPENBHB, index=False)
    plot_age_distribution(df_openbhb, "open mind + hcp + openbhb")

def print_info_ages(df):

    # 1. Create larger bins (5-year bins)
    bin_width = 5
    bins = np.arange(df["age"].min()//bin_width*bin_width, df["age"].max()+bin_width, bin_width)
    labels = [f"{int(bins[i])}-{int(bins[i+1]-1)}" for i in range(len(bins)-1)]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # 2. Map sites: hcp, open_mind, others → openbhb
    df["site_group"] = df["site"].map({"hcp":"hcp", "open_mind":"open_mind"}).fillna("openbhb")

    # 3. Count number of subjects per bin per site
    count_table = df.groupby(["age_bin", "site_group"]).size().unstack(fill_value=0)

    # 4. Plot
    ax = count_table.plot(
        kind="bar", 
        stacked=False, 
        figsize=(12,6),
        color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )

    # 5. Annotate bars with counts
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width()/2
        ax.text(x, height + 1, str(int(height)), ha='center', fontsize=18, rotation=90)

    plt.xlabel("Age bin (years)", fontsize=24)
    plt.ylabel("Number of subjects", fontsize=24)
    plt.title(f"Number of subjects per {bin_width}-year age bin by site", fontsize=26)
    plt.xticks(rotation=45)
    plt.legend(title="Site", fontsize=20)
    plt.tight_layout()
    plt.show()


def main():
    save_openBHB_dataframe() # to save OpenBHB roi dataframe if it doesn't already exist

    # for OpenBHB aggregated with other cohorts
    # create_df(False)
    df = pd.read_csv(DF_PATH_WITH_OPENBHB)
    print_info_ages(df)

if __name__ == "__main__":
    main()
