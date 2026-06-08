
################################################################################
# %% 1. Initialization 
# ====================
import os.path
from datetime import datetime
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from config import config, cv_val
from ml_utils import PredefinedSplit, group_by_roi_fs
from mulm.residualizer.residualizer import Residualizer, ResidualizerEstimator
from utils.sklearn_utils import classification_report_cv, drop_indices_from_folds
from utils.pandas_utils import safe_merging_toref

################################################################################
# %% CONFIGURATION
# ================

INPUT_CAT12_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'
INPUT_FS_DATA = './data/processed/fs_aseg_volumes.tsv'

OUTPUT = "reports/classification_fs_reports.xlsx"
cv_test = PredefinedSplit(json_file=config['cv_test'])


################################################################################
# %% Load Data and Merge with CAT12 data
# ============

def load_vbm_data(features = []):
    data_cat12 = pd.read_csv(INPUT_CAT12_DATA, index_col="participant_id")[config['residualization']+["y"] + features]
    return data_cat12


def load_fs_data(features = []):
    data_fs = pd.read_csv(INPUT_FS_DATA, index_col="participant_id", sep="\t")
    data_fs = data_fs[data_fs.session == "M00"]
    data_fs.drop(columns=["session"], inplace=True)
    data_fs = data_fs[features]
    return data_fs


def merge_datasets(data_vbm, data_fs):
    """
    Merge the VBM and FS datasets.
    Return
    ------
    data : merged DataFrame with NaN for missing subjects in new (FS)
    data_merged : merged DataFrame before dropping NaN, with participant_id index for reference
    na_index : index of the row with NaN (missing from new)
    """
    
    # Merge the datasets using the safe_merging_toref function
    data = safe_merging_toref(data_vbm, data_fs, how="left", verbose=True, add_flags=False)
    """
    ────────────────────────────────────────────────────────────
    PRE-MERGE DIAGNOSTICS
    ────────────────────────────────────────────────────────────
    ref  subjects          :    117
    new  subjects          :    135
    Common (will merge)    :    116

    ⚠  In ref but NOT in new (1) → new columns will be NaN:
        [  35]  sub-41252

    ℹ  In new but NOT in ref (19) → will be appended at the end:
                sub-14530
                sub-16345
                sub-16572
                sub-19151
                sub-21920
                sub-25791
                sub-26675
                sub-28390
                sub-47689
                sub-48868
                sub-51499
                sub-54766
                sub-64123
                sub-65385
                sub-69513
                sub-84381
                sub-87329
                sub-97551
                sub-98297
    ────────────────────────────────────────────────────────────

    Merged DataFrame : 117 rows × 51 columns
    → 1 row(s) have NaN in new columns (missing from new)
    """

    assert np.all(data.index == data_vbm.index)  # Check that the index is in the same order as the ref
    # (i) indices of rows containing at least one NaN
    na_participants = data[data.isna().any(axis=1)].index
    assert na_participants == "sub-41252"
    
    # Prepare to drop the row with NaN (missing from new) and reset index
    data_merged = data.copy()
    data.reset_index(drop=True, inplace=True)  # Drop the index to avoid issues with models
    na_index = data[data.isna().any(axis=1)].index
    assert na_index == 35
    data = data.drop(index=na_index)
    data.reset_index(drop=True, inplace=True)  # Reset index after dropping the row
    assert data.shape[0] == 116

    return data, data_merged, na_index  # Return both the cleaned data and the merged data with NaN and participant_id index for reference


def get_residualizer(data, X, formula_res):

    from mulm.residualizer import Residualizer, ResidualizerEstimator

    res = Residualizer(data=data, formula_res=formula_res)

    # Extract design matrix and pack it with X
    Z = res.get_design_mat(data=data)
    res_estimator = ResidualizerEstimator(res)
        
    # Pack Z with X
    ZX = res_estimator.pack(Z, X)
    Z_, X_ = res_estimator.upack(ZX)
    assert np.all(X_ == X)
    assert Z.shape[1] + len(feature_columns) == ZX.shape[1]  # Check that the number of columns is correct

    # Finally, we will use ZX as X for the models, and Z as the residualization part
    #X = ZX
    
    return res, res_estimator, ZX

def make_classification(X, y, cv_test, models):
    """
    Run classification models using cross_validate and compute the specified scorers.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target labels.
    cv_test : cross-validation strategy
        The cross-validation splitting strategy to use.
    models : dict
        A dictionary of model name to model instance.

    Returns
    -------
    metrics_df : DataFrame
        A DataFrame containing the average metrics for each model.
    """

    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score, roc_auc_score

    # Scorers including recall for each class
    recall_scorers = {
        f"recall_class_{label}": make_scorer(recall_score, labels=[label], average="macro")
        for label in np.unique(y)
    }

    scorers = {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        **recall_scorers,
    }

    metrics = ['test_%s' % metric for metric in scorers.keys()]

    def average_metrics(cv_res, metrics):
        """Average metrics from cross-validation results."""
        # Extract the metrics from the cv_res dictionary
        # e.g., cv_res = {'test_balanced_accuracy': [...], 'test_roc_auc': [...], ...}
        # Return the mean of each metric
        return [cv_res[metric].mean() for metric in metrics]

    # Evaluate each model using cross-validation
    # and stack the average metrics

    rows = []
    rows_historical = []

    for mod, model in models.items():
        print("-" * 80)
        print(f"Evaluating model: {mod}")

        cv_res = cross_validate(
            estimator=model, X=X, y=y,
            cv=cv_test, scoring=scorers,
            return_estimator=True,
            return_train_score=False, n_jobs=5, verbose=50)

        metrics_row_historical = average_metrics(cv_res, metrics=['test_balanced_accuracy', 'test_roc_auc', 'test_recall_class_0', 'test_recall_class_1'])
        metrics_row = classification_report_cv(X, y, cv_res['estimator'],
                                            cv=cv_test, as_one_row=True)
        metrics_row.index = [mod]
        rows.append(metrics_row)
        rows_historical.append([mod] + metrics_row_historical)

    metrics_df = pd.concat(rows)
    #print(metrics_df)

    metrics_historical_df = pd.DataFrame(rows_historical, columns=['model', 'test_balanced_accuracy', 
                                                                'test_roc_auc',
                                                                'test_recall_class_0', 'test_recall_class_1'])
    #print(metrics_historical_df)
    
    return metrics_df

# %%
#===============================================================================
# Main execution
#===============================================================================

# %% Load and merge data
data_vbm = load_vbm_data(features = ["Left Amygdala_GM_Vol", "Left Hippocampus_GM_Vol"])
data_vbm = load_vbm_data()
data_fs = load_fs_data(features = ["Left-Amygdala", "Left-Hippocampus", "Right-Amygdala", "Right-Hippocampus"])
data_fs = load_fs_data(features = ["Left-Amygdala", "Left-Hippocampus"])

data, data_merged, na_index = merge_datasets(data_vbm, data_fs)

# %% Select Input = dataframe - (target + drop + residualization)
feature_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]
X = data[feature_columns].values
y = data[config['target']].map(config['target_remap'])

# %% Drop indices from cv_test if needed
from utils.sklearn_utils import drop_indices_from_folds
cv_test = PredefinedSplit(json_file=config['cv_test'])
cv_test.predefined_splits = drop_indices_from_folds(cv_test.predefined_splits, drop_indices=na_index)

# %% Residualization if needed
if config['residualization']:
    columns_res = config['residualization']
    formula_res = "+".join(columns_res)
    res, res_estimator, X = get_residualizer(data, X, formula_res)

from ml_utils import group_by_roi_fs
roi_groups = group_by_roi_fs(feature_columns)
# Convert roi_groups to indices
roi_groups = {roi:[feature_columns.index(x) for x in cols] for roi, cols in roi_groups.items()}

# %% Models
from ml_utils import make_models

models = make_models(n_jobs_grid_search=5, cv_val=cv_val,
                    residualization_formula=formula_res,
                    residualizer_estimator=res_estimator,
                    roi_groups=roi_groups)

# Debug with onlys one model
if False:
    k = 'model-grpRoiLda+lrl2_resid-age+sex+site'
    models = {k:models[k]}

del models['model-grpRoiLda+lrl2_resid-age+sex+site']

# %% Run classification and save results

metrics_df = make_classification(X, y, cv_test, models)

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    metrics_df.to_excel(writer,            sheet_name="metrics")
    # metrics_historical_df.to_excel(writer, sheet_name="metrics_historical", index=False)
print(f"\n✔  Saved {OUTPUT}")

print(metrics_df)

"""
metric                            Balanced Accuracy                                                       ROC-AUC                                ... Recall (class 1)                  F1                                     MCC                              
                                          avg_folds std_folds  se_folds    pooled pval_pooled pval_fold avg_folds std_folds  se_folds    pooled  ...      pval_pooled pval_fold avg_folds std_folds  se_folds    pooled avg_folds std_folds  se_folds    pooled
model-lrl2cv_resid-age+sex+site            0.613611  0.077292  0.034566  0.611969    0.016136  0.021199  0.640000  0.118720  0.053093  0.654440  ...         0.044215  0.053466  0.537716  0.080559  0.036027  0.540000  0.219453  0.148894  0.066587  0.215249
model-lrenetcv_resid-age+sex+site          0.500000  0.000000  0.000000  0.500000    0.536961       NaN  0.500000  0.000000  0.000000  0.500000  ...         1.000000  1.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
model-svmrbfcv_resid-age+sex+site          0.482500  0.064893  0.029021  0.478443    0.742043  0.690867  0.355106  0.092087  0.041182  0.414736  ...         0.082075  0.221124  0.422317  0.122191  0.054645  0.444444 -0.033691  0.127788  0.057148 -0.043345
model-forestcv_resid-age+sex+site          0.472698  0.113780  0.050884  0.472329    0.798265  0.671821  0.552037  0.128933  0.057660  0.550837  ...         0.996042  0.900602  0.292063  0.185273  0.082857  0.317073 -0.061153  0.242432  0.108419 -0.055957
model-gbcv_resid-age+sex+site              0.518532  0.068020  0.030419  0.518018    0.390370  0.307392  0.549365  0.095622  0.042763  0.545528  ...         0.990240  0.937917  0.344238  0.121627  0.054393  0.358974  0.036663  0.142154  0.063573  0.037435
model-mlp_cv_resid-age+sex+site            0.602183  0.088217  0.039452  0.600708    0.025365  0.040719  0.652963  0.161866  0.072389  0.656049  ...         0.996042  0.949771  0.397143  0.166479  0.074452  0.412698  0.246413  0.182854  0.081775  0.251398
"""