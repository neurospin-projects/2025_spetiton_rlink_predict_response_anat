import pandas as pd, numpy as np
from utils import make_stratified_splitter
from collections import defaultdict

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"

seed=11 # seed=11 gives ROC-AUC = 69% (mean), and highest balanced accuracy over train set
str_WM=""
df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00"+str_WM+"_v4labels.csv")
df_ROI_age_sex_site["y"] = df_ROI_age_sex_site["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
df_ROI_age_sex_site = df_ROI_age_sex_site.reset_index(drop=True)


# To track how often each site is in test sets
site_test_counts = defaultdict(int)
df = df_ROI_age_sex_site.copy()
prop_GR_tot = round(df['y'].mean(),2) 
# print("proportion GR total ",prop_GR_tot)

print(df["site"].value_counts())
print("\n\n")
for fold_idx, (train_idx, test_idx) in enumerate(make_stratified_splitter(df,cv_seed=seed)):
    print(f"\n=== Fold {fold_idx + 1} ===")
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    train_sites = set(df_train["site"])
    test_sites = set(df_test["site"])
    all_sites = set(df["site"])
    train_age_mean = round(df_train["age"].mean(),2)
    test_age_mean = round(df_test["age"].mean(),2)
    prop_female_tr = round(df_train['sex'].mean(),2) 
    prop_female_te = round(df_test['sex'].mean(),2)
    prop_GR_tr = round(df_train['y'].mean(),2) 
    prop_GR_te = round(df_test['y'].mean(),2)

    missing_in_train = all_sites - train_sites
    missing_in_test = all_sites - test_sites

    print(f"Train sites ({len(train_sites)}): {sorted(train_sites)}")
    print(f"Test sites ({len(test_sites)}): {sorted(test_sites)}")
    print(f"❗ Missing in train: {sorted(missing_in_train)}")
    print(f"❗ Missing in test: {sorted(missing_in_test)}")
    print("Train age mean ",train_age_mean)
    print("Train proportion female ", prop_female_tr)
    print("Train proportion GR ", prop_GR_tr)

    print("Test age mean ",test_age_mean)
    print("Test proportion female ", prop_female_te)
    print("Test proportion GR ", prop_GR_te)

    # Count how often each site is seen in test sets
    for site in test_sites:
        site_test_counts[site] += 1

# After folds: summarize site test presence
print("\n=== Site Test Frequency Summary ===")
for site in sorted(site_test_counts):
    print(f"Site {site}: in {site_test_counts[site]} of 5 test sets")
