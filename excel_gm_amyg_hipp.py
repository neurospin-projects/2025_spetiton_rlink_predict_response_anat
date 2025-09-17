import pandas as pd, numpy as np, os

ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
DF_ROI_M3M0 = DATA_DIR+"df_ROI_age_sex_site_M00_M03_v4labels.csv" 

df_ROI_age_sex_site = pd.read_csv(DF_ROI_M3M0)
print(df_ROI_age_sex_site["sex"].unique())


print(df_ROI_age_sex_site)
four_rois = ["Left Hippocampus_GM_Vol", "Right Hippocampus_GM_Vol","Right Amygdala_GM_Vol", "Left Amygdala_GM_Vol"]

columns_selected = [r for r in list(df_ROI_age_sex_site.columns) if r in four_rois or r in \
                    ["participant_id","age","sex","site","response","session"]]
df_ROI_age_sex_site=df_ROI_age_sex_site[columns_selected].reset_index(drop=True)

# checkups
# dup_mask = (
#     df_ROI_age_sex_site.groupby("participant_id")[["age", "sex", "site", "y"]]
#       .nunique()
#       .gt(1)   # True if > 1 unique value within the same participant
# )

# # participants with inconsistencies
# bad_participants = dup_mask.any(axis=1)
# print("issue with :" ,df_ROI_age_sex_site[df_ROI_age_sex_site["participant_id"].isin(bad_participants[bad_participants].index)])

dict_M00= {roi:roi+"_M00" for roi in four_rois}
dict_M03= {roi:roi+"_M03" for roi in four_rois}

df_ROI_age_sex_siteM00 = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M00"].reset_index(drop=True)
df_ROI_age_sex_siteM03 = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M03"].reset_index(drop=True)

df_ROI_age_sex_siteM00 = df_ROI_age_sex_siteM00.rename(columns=dict_M00).drop(columns="session")
df_ROI_age_sex_siteM03 = df_ROI_age_sex_siteM03.rename(columns=dict_M03).drop(columns="session")

print(df_ROI_age_sex_siteM00)
print(df_ROI_age_sex_siteM03)
new_df = pd.merge(df_ROI_age_sex_siteM00, df_ROI_age_sex_siteM03, on=["participant_id","age","sex","site","response"], how="inner")

# reorder columns
cols_to_move = ['age', 'sex', 'site', 'response']
new_df = new_df[[c for c in new_df.columns if c not in cols_to_move]+cols_to_move]

print(new_df)
print(list(new_df.columns))

for roi in four_rois:
    rows = new_df[new_df[roi+"_M00"] < new_df[roi+"_M03"]]
    print(roi," ",len(rows))

new_df.to_excel(ROOT+"250908_GM_vol_AmygHip_M0M3.xlsx", index=False)  
