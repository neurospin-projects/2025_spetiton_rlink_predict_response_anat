import numpy as np, pandas as pd
from utils import read_pkl

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
RESULTS_DIR = ROOT+"reports/classification_results/"
FOLDS_DIR = ROOT+"reports/folds_CV/"

data = read_pkl(RESULTS_DIR+"results_classification_seed_1_GRvsPaRNR_5fold_17rois_only.pkl")
data=data[data["residualization"]=="res_age_sex_site"]
data=data[data["classifier"]=="L2LR"]
folds_subjects = read_pkl(FOLDS_DIR+"subjects_for_each_fold_GRvsPaRNR_5foldCV_seed_1.pkl")


ytest_all_folds , ypred_all_folds, test_subjects =  [], [], []

for fold in range(5):
    ytest_all_folds.append(data[data["fold"]==fold]["y_test"].values[0])
    print(np.shape(data[data["fold"]==fold]["y_test"].values[0]))
    ypred_all_folds.append(data[data["fold"]==fold]["y_pred_test"].values[0])
    test_subjects.append(folds_subjects[fold]["test_subjects_ids"])


test_subjects = np.concatenate(test_subjects,axis=0)
ytest_all_folds = np.concatenate(ytest_all_folds,axis=0)
ypred_all_folds = np.concatenate(ypred_all_folds,axis=0)
print(np.shape(ypred_all_folds), np.shape(ytest_all_folds))
print("test_subjects", np.shape(test_subjects))
print(np.unique(ypred_all_folds))
df=pd.DataFrame({"participant_id":test_subjects, "y_pred":ypred_all_folds, "y_test":ytest_all_folds})
print(df)
incorrect_classifications = df[df['y_pred'] != df['y_test']]
print(len(incorrect_classifications)," subjects out of "+str(len(df))+" are incorrectly classified")
# print(df[df["participant_id"]=="sub-41252"])

rois_df = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
print(rois_df)

incorrect_classif_info = pd.merge(incorrect_classifications, rois_df, how="inner", on="participant_id")
print(incorrect_classif_info[["participant_id","age","sex","site","y"]])
print("nb of partial responders incorrectly classified ",\
      len(incorrect_classif_info[incorrect_classif_info["y"]=="PaR"])) # 16
print("nb of non responders incorrectly classified ",\
      len(incorrect_classif_info[incorrect_classif_info["y"]=="NR"])) # 7
print("nb of good responders incorrectly classified ",\
      len(incorrect_classif_info[incorrect_classif_info["y"]=="GR"])) # 10