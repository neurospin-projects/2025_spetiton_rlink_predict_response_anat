from utils import read_pkl
import numpy as np, pandas as pd

path = "reports/classification_results/"


def get_metrics_res(df, res="res_age_sex_site"):

    return df[df["residualization"]==res]["roc_auc_test"].mean(),\
    df[df["residualization"]==res]["roc_auc_test"].std(),\
    df[df["residualization"]==res]["balanced_accuracy_test"].mean(),\
    df[df["residualization"]==res]["balanced_accuracy_test"].std()

def fill_dict(res_dict, df, whitened=False):
    for res_loop in ["res_age_sex_site", "res_age_sex", "no_res"]:
        mean_roc_auc , std_roc_auc ,  mean_bacc, std_bacc =\
        get_metrics_res(df, res = res_loop)

        res_dict.append({"seed":seed, "mean_roc_auc": mean_roc_auc, "std_roc_auc":std_roc_auc,\
        "residualization": res_loop, "whitened":whitened, "mean_bacc": mean_bacc, "std_bacc": std_bacc})

res = []

for seed in range(1,101):
    data_whitened = read_pkl(path+"results_classification_GRvsPaRNR_5fold_v4labels_whitened_24juin25_"+str(seed)+"cvseed.pkl")
    data = read_pkl(path+"results_classification_GRvsPaRNR_5fold_v4labels_24juin25_"+str(seed)+"cvseed.pkl")
    fill_dict(res, data, whitened=False)
    fill_dict(res, data_whitened, whitened=True)

df_res = pd.DataFrame(res)
residualization ="res_age_sex_site"
print("results dataframe :",df_res)
df_selected = df_res[
    (df_res["residualization"] == residualization) & 
    (df_res["whitened"] == True)
]
print(" mean of columns for residualization on age, sex, and site with withening \n",df_selected.mean(numeric_only=True))
df_selected = df_res[
    (df_res["residualization"] == residualization) & 
    (df_res["whitened"] == False)
]
print(" mean of columns for residualization on age, sex, and site WITHOUT withening \n",df_selected.mean(numeric_only=True))    

df_sub = df_res[df_res["residualization"] == residualization]
pivot_roc_auc = df_sub.pivot(index="seed", columns="whitened", values="mean_roc_auc")
pivot_bacc =  df_sub.pivot(index="seed", columns="whitened", values="mean_bacc")
# Compute the pairwise difference for each seed (True - False)
pivot_roc_auc["difference"] = pivot_roc_auc[True] - pivot_roc_auc[False]
print(pivot_roc_auc)
pivot_bacc["difference"] = pivot_bacc[True] - pivot_bacc[False]

# Compute the mean of these differences
mean_difference_rocauc = pivot_roc_auc["difference"].mean()
mean_difference_bacc = pivot_bacc["difference"].mean()

print("Mean ROC AUC test difference (whitened - unwhitened): ", mean_difference_rocauc)
print("Mean Balanced Accuracy test difference (whitened - unwhitened): ", mean_difference_bacc)
