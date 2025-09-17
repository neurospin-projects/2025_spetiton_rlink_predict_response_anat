import pandas as pd
import matplotlib.pyplot as plt
from utils import get_lists_roi_in_both_openBHB_and_rlink
import numpy as np
import matplotlib.pyplot as plt

# inputs
PARTICIPANTS_FILE_OPENMIND = "/neurospin/psy/openmind/participants.tsv"
ROI_FILE_OPENMIND = "/neurospin/psy/openmind/derivatives/cat12-12.8.2_vbm_roi/neuromorphometrics_cat12_vbm_roi.tsv"
ROI_HCP_FILE = "/neurospin/psy/hcp/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
PARTICIPANTS_HCP = "/neurospin/psy/hcp/HCP_t1mri_mwp1_participants_v-2023.csv" #/neurospin/psy/hcp/participants.tsv"
OPENBHB_DATAFRAME = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/data/processed/OpenBHB_roi.csv"
ROIS_UKB = "/neurospin/psy/ukb/derivatives/soft-cat12_provider-ns_ver-7.r1743_vbmroi/cat12_vbm_roi.tsv"
UKB_PARTICIPANTS="/neurospin/psy/ukb/UKB_t1mri_mwp1_participants.csv"
UKB_PHENOTYPE="/neurospin/psy/ukb/phenotypes/ses-0/baseline-characteristics-population-characteristics.tsv"
# output
DF_PATH_WITH_OPENBHB = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/data/processed/hcp_open_mind_ukb_openbhb_roi_vbm_with_age.csv"

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

def get_ukb(verbose=False):
    df = pd.read_csv(ROIS_UKB, sep="\t")
    participants=pd.read_csv(UKB_PARTICIPANTS, sep="\t")
    phenotype = pd.read_csv(UKB_PHENOTYPE, sep="\t")

    phenotype = phenotype.rename(
        columns={
            "31": "sex", # 0 is female and 1 is male
            "34": "year_of_birth",
            "52": "month_of_birth",
            "189": "townsend_index",
            "21022": "age" #"age_at_recruitment",
        }
    )

    phenotype["participant_id"] = "sub-" + phenotype["participant_id"].astype(str)
    participants["participant_id"] = "sub-" + participants["participant_id"].astype(str)
    df = df.drop(columns=["Unnamed: 0"])
    df_merged=pd.merge(df, phenotype, on="participant_id", how="inner")

    # remove sessions of participants that have multiple sessions that aren't the first ones (since we have age a recruitment)
    df_merged = (
        df_merged.sort_values(["participant_id", "session"])
        .drop_duplicates("participant_id", keep="first")
        .reset_index(drop=True)
    )

    assert df_merged["participant_id"].is_unique   # make sure there are no participant_id duplicates

    if verbose: print(df_merged.groupby("sex")["tiv"].mean()) # checking mean tiv by sex label to confirm men are labelled 1 and women 0
    df_merged["sex"] = df_merged["sex"].replace({0: 1, 1: 0}) # switch it
    if verbose: print(df_merged.groupby("sex")["tiv"].mean()) # checking it got switched


    rois = [r for r in list(df_merged.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
    df_merged = df_merged[["participant_id"]+rois+["tiv","sex","age"]]

    # scale by TIV such that TIV is equal to 1500.0
    target_tiv = 1500.0
    scaling_factor = target_tiv / df_merged["tiv"]
    df_merged[rois+["tiv"]] = df_merged[rois+["tiv"]].mul(scaling_factor, axis=0)
    # plot_age_distribution(df_merged, "ukb")

    return df_merged

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
    rois_open_mind["site"]=rois_open_mind["dataset"].astype(str)
    rois_open_mind["participant_id"] = rois_open_mind["dataset"].astype(str) + "__" + rois_open_mind["participant_id"].astype(str)
    # list of rois (GM and CSF volumes)
    all_rois_open_mind = [ r for r in list(rois_open_mind.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]

    # 2. add "age" column to df
    participants_open_mind["sex"] = participants_open_mind["sex"].map({"female": 1, "male": 0})
    rois_open_mind = pd.merge(rois_open_mind, participants_open_mind[["participant_id","age","sex"]], on="participant_id", how="inner")
    list_roi_openbhb, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    # print(len(list_roi_rlink), len(list_roi_openbhb)) # 252 for both
    # print(rois_open_mind.groupby("sex")["tiv"].mean()) # checking mean tiv by sex label to confirm men are labelled 0 and women 1

    # 3. scale by TIV such that TIV is equal to 1500.0
    target_tiv = 1500.0
    scaling_factor = target_tiv / rois_open_mind["tiv"]
    rois_open_mind[all_rois_open_mind+["tiv"]] = rois_open_mind[all_rois_open_mind+["tiv"]].mul(scaling_factor, axis=0)


    # 4. rename rois to match openbhb roi names
    dict_rlink_to_openbhb_roi_names = dict(zip(list_roi_rlink, list_roi_openbhb)) # get dict of roi names correspondencies
    rois_open_mind = rois_open_mind[list_roi_rlink+["age","sex","site"]]
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
    
    rois_hcp = pd.merge(rois_hcp, participants_hcp[["participant_id","age","sex","site"]], on="participant_id", how="inner")
    # print(rois_hcp.groupby("sex")["TIV"].mean()) # checking mean tiv by sex label to confirm men are labelled 0 and women 1

    # 2. scale by TIV such that TIV is equal to 1500.0
    target_tiv = 1500.0
    scaling_factor = target_tiv / rois_hcp["TIV"]
    rois_hcp[all_rois_hcp+["TIV"]] = rois_hcp[all_rois_hcp+["TIV"]].mul(scaling_factor, axis=0)

    # 3. add "age" column to df
    # plot_age_distribution(rois_hcp, "hcp")
    rois_hcp = rois_hcp[list_roi_openbhb+["age","sex","site"]]

    # 4. add cohort column
    rois_hcp["cohort"]="hcp"
    print(rois_hcp)

    # ============== UKB ======================
    rois_ukb = get_ukb()
    rois_ukb = rois_ukb[list_roi_openbhb+["age","sex"]]
    rois_ukb["cohort"]="ukb"
    rois_ukb["site"]="ukb" # 42923 subjects

    # concatenate
    df_all = pd.concat([rois_hcp, rois_open_mind], axis=0).reset_index(drop=True)
    df_all = pd.concat([df_all, rois_ukb], axis=0).reset_index(drop=True)  #54200 subjects at this point

    # plot_age_distribution(df_all, "open mind + hcp")
    df_openbhb = pd.read_csv(OPENBHB_DATAFRAME)
    df_openbhb = df_openbhb[list_roi_openbhb+["age","sex","site"]]
    df_openbhb["cohort"]="openbhb" #6247 subjects

    df_all = pd.concat([df_all, df_openbhb], axis=0).reset_index(drop=True)
    print(df_all)

    df_all["sex"] = df_all["sex"].map({1:"female", 0:"male"})  # re-label "sex" to "female" and "male"
    df_all = df_all.dropna() # remove subjects with missing data

    if save: df_all.to_csv(DF_PATH_WITH_OPENBHB, index=False) # 60447 subjects
    plot_age_distribution(df_all, "open mind + hcp + openbhb + ukb")

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
    create_df(True)
    quit()
    df = pd.read_csv(DF_PATH_WITH_OPENBHB)
    print_info_ages(df)

if __name__ == "__main__":
    main()


    """
    aggréger à OpenBHB
    sampler les participants en fct du nb de sujets à chaque âge (enfin à chaque bin)

    """