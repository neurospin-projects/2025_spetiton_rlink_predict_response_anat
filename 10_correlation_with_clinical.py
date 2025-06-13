import pandas as pd 
import sys, os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import re

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR_CLINICAL = "/neurospin/signatures/2025_spetiton_rlink_predict_response_clinical/data/processed/"
DATA_DIR=ROOT+"data/processed/"
DF_ROI_M0=DATA_DIR+"df_ROI_age_sex_site_M00_v4labels.csv"

clinical_df = pd.read_csv(DATA_DIR_CLINICAL+"clinical_df.csv")


roi_df_m0 = pd.read_csv(DF_ROI_M0)
print(roi_df_m0)
merge = pd.merge(roi_df_m0[["y","participant_id"]], clinical_df, on="participant_id", how="inner")
merge["sex"] = merge["sex"].replace({2: 1, 1: 0})
merge["y"] = merge["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
print(merge["AGESTBH2_PLI"])
print(merge["AGESTBH2_PLI"])
# print(merge["AGEONSET"])

merge["duration of illness"]= merge["AGENDBD2_PLI"]-merge["AGESTBH2_PLI"]
nan_count = merge['AGESTBH2_PLI'].isna().sum()
print(nan_count)
nan_count = merge['AGESTBH2_PLI'].isna().sum()
print(nan_count)

print(merge)

# compute correlation of response with clinical data
correlations = merge.drop(columns=['participant_id']).corr(numeric_only=True)
# drop(y) to remove correlation of y to itself
print(correlations)
high_corrs = correlations["y"][correlations["y"].abs() > 0.3]
print(high_corrs)