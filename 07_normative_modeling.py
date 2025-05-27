import numpy as np, pandas as pd
from utils import scale_rois_with_tiv

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"

atlas= DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv"

# find roi openbhb 5k
path_to_data = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/"

# no abide2 roi
# 15 datasets

# paths
biobd_bsnip_roi = pd.read_csv("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/VBMROI_Neuromorphometrics.csv")
biobd_bsnip_participants = pd.read_csv("/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/participantsBD.csv")

adni_roi = path_to_data+"ADNI_cat12vbm_mwp1_roi.tsv"
adni_participants = path_to_data+"adni-1st-session_mwp1_participants.csv"

oasis_participants = path_to_data+"oasis3_t1mri_mwp1_participants.csv"
oasis_roi = "/neurospin/psy_sbox/oasis3/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"

icbm_participants = path_to_data+"icbm_t1mri_mwp1_participants.csv"
icbm_rois = "/neurospin/psy_sbox/icbm/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"

hcp_participants = path_to_data+"hcp_t1mri_mwp1_participants.csv"
hcp_roi = "/neurospin/psy_sbox/hcp/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"

scz_participants = path_to_data+"schizconnect-vip_t1mri_mwp1_participants.csv"
scz_roi = "/neurospin/psy/schizconnect-vip-prague/derivatives/cat12-12.7_vbm_roi/cat12-12.7_vbm_roi.tsv"

abide1_roi = "/neurospin/psy_sbox/abide1/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
abide1_participants = path_to_data+"abide1_t1mri_mwp1_participants.csv"

ixi_roi = "/neurospin/psy_sbox/ixi/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
ixi_participants = path_to_data+"ixi_t1mri_mwp1_participants.csv"

npc_roi = "/neurospin/psy_sbox/npc/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
npc_participants = path_to_data+"npc_t1mri_mwp1_participants.csv"

rbp_roi="/neurospin/psy_sbox/rbp/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
rbp_participants = path_to_data+"rbp_t1mri_mwp1_participants.csv"

gsp_roi="/neurospin/psy_sbox/GSP/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
gsp_participants = path_to_data+"gsp_t1mri_mwp1_participants.csv"

localizer_roi="/neurospin/psy_sbox/localizer/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
localizer_participants = path_to_data+"localizer_t1mri_mwp1_participants.csv"

mpi_roi="/neurospin/psy_sbox/mpi-leipzig/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
mpi_participants = path_to_data+"mpi-leipzig_t1mri_mwp1_participants.csv"

corr_roi="/neurospin/psy_sbox/CoRR/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
corr_participants = path_to_data+"corr_t1mri_mwp1_participants.csv"

nar_roi="/neurospin/psy_sbox/nar/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
nar_participants = path_to_data+"nar_t1mri_mwp1_participants.csv"
"""
oasis : choose session
ICBM : same thing
ADNI : same thing
GSP: same thing
MPI
CoRR
NAR
"""


def get_df_one_study_healthy_controls(path_roi, path_participants):
    # atlas (Neuromorphometrics) dataframe
    atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")
    list_roi_abbr = list(atlas_df["ROIabbr"].values) # roi abbreviations 

    participants_df = pd.read_csv(path_participants)
    participants_df = participants_df[participants_df["diagnosis"]=="control"] # keep only controls
    roi_data = pd.read_csv(path_roi, sep="\t")
    roi_df = pd.merge(roi_data , participants_df[["participant_id","age","sex","site"]], on="participant_id")

    if "session" in roi_df.columns:
        if np.array_equal(np.sort(roi_df["session"].unique()), [1, 2]):
            roi_df = roi_df[roi_df["session"] == 1]

    roi_volumes = [roi for roi in list(roi_df.columns) if roi.endswith("_CSF_Vol") or roi.endswith("_GM_Vol")]
    counts = roi_df["participant_id"].value_counts()
    duplicates = counts[counts > 1]
    
    if not duplicates.empty:
        print(duplicates)

        print(participants_df.columns)
        print(roi_df["session"].unique())
        print(participants_df["session"].unique())
        print(participants_df["study"].unique())
        
        quit()


    if "tiv" in roi_df.columns: roi_df = roi_df[roi_volumes+["participant_id","age","sex","site","tiv"]]
    if "TIV" in roi_df.columns: roi_df = roi_df[roi_volumes+["participant_id","age","sex","site","TIV"]]
 
    roi_df = scale_rois_with_tiv(roi_df, roi_volumes, target_tiv=1500.0) # scale TIV to 1500.0
    assert len(roi_volumes)==len([roi for roi in roi_volumes if roi in list_roi_abbr]),\
    "the df of oasis database rois doesn't have the same abbreviations as the atlas dataframe"
    
    return roi_df

# schizconnect-vip database
scz_df = get_df_one_study_healthy_controls(scz_roi, scz_participants)

# hcp database
hcp_df = get_df_one_study_healthy_controls(hcp_roi, hcp_participants)

# oasis3 database
# oasis_df = get_df_one_study_healthy_controls(oasis_roi, oasis_participants)

# icbm database
print("ICBM")
# icbm_df = get_df_one_study_healthy_controls(icbm_rois, icbm_participants)

atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")
list_roi_abbr = list(atlas_df["ROIabbr"].values) #roi abbreviations 
# biobd bsnip databases
biobd_bsnip_roi_volumes = [roi for roi in list(biobd_bsnip_roi.columns) if roi.endswith("_CSF_Vol") or roi.endswith("_GM_Vol")]
biobd_bsnip_participants = biobd_bsnip_participants[biobd_bsnip_participants["diagnosis"]=="control"]#  keep only controls
biobd_bsnip_df = pd.merge(biobd_bsnip_roi, biobd_bsnip_participants[["participant_id","age","sex","site"]], on="participant_id")
biobd_bsnip_df = biobd_bsnip_df[biobd_bsnip_roi_volumes+["participant_id", "age","sex","site","TIV"]]
assert len(biobd_bsnip_roi_volumes)==len([roi for roi in biobd_bsnip_roi_volumes if roi in list_roi_abbr]),\
    "the df of oasis database rois doesn't have the same abbreviations as the atlas dataframe"

print("ADNI")
# adni database
# adni_df = get_df_one_study_healthy_controls(adni_roi, adni_participants)

print("ABIDE")
# abide database
abide_df = get_df_one_study_healthy_controls(abide1_roi, abide1_participants)

print("IXI")
# ixi database
ixi_df = get_df_one_study_healthy_controls(ixi_roi, ixi_participants)

print("NPC")
# npc database
npc_df = get_df_one_study_healthy_controls(npc_roi, npc_participants)

print("rbp")
# rbp database
rbp_df = get_df_one_study_healthy_controls(rbp_roi, rbp_participants)

print("gsp")
# gsp database
# gsp_df = get_df_one_study_healthy_controls(gsp_roi, gsp_participants)

print("localizer")
# localizer database
localizer_df = get_df_one_study_healthy_controls(localizer_roi, localizer_participants)

print("MPI")
# mpi database
# mpi_df = get_df_one_study_healthy_controls(mpi_roi, mpi_participants)

print("Corr")
# corr database
# corr_df = get_df_one_study_healthy_controls(corr_roi, corr_participants)

print("nar")
# nar database
nar_df = get_df_one_study_healthy_controls(nar_roi, nar_participants)



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
    counts = df["participant_id"].value_counts()
    duplicates = counts[counts > 1]
    print("\n\ncpt:",cpt, names_databases[cpt], " \nduplicates ",duplicates)
    cpt+=1

df_concat = pd.concat(list_df, axis=0, ignore_index=True)
print(df_concat)
counts = df_concat['participant_id'].value_counts()
duplicates = counts[counts > 1]

print(duplicates)

