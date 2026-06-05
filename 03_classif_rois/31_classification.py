'''
Supervised Classification: ROIs with Repeated CV of many models
===============================================================

rsync -azvu --exclude study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4_task-predLiResp/20_ml-classifLiResp.cachedir study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4_task-predLiResp triscotte.intra.cea.fr:/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/edouard/

'''

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
from ml_utils import PredefinedSplit
from mulm.residualizer.residualizer import Residualizer, ResidualizerEstimator
from utils.sklearn_utils import classification_report_cv
from ml_utils import drop_indices_from_folds

################################################################################
# %% CONFIGURATION
# ================

INPUT_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'
OUTPUT = "reports/classification_reports.xlsx"
cv_test = PredefinedSplit(json_file=config['cv_test'])


################################################################################
# %% Load Data
# ============

data = pd.read_csv(INPUT_DATA)
assert data.shape == (117, 273)

# Select Input = dataframe - (target + drop + residualization)
feature_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]

X = data[feature_columns].values
y = data[config['target']].map(config['target_remap'])

# Multiply CSF columns by -1
csf_indices = np.array([i for i, col in enumerate(feature_columns) if 'CSF' in col])
X[:, (csf_indices)] *= -1

assert X.shape[1] == len(feature_columns)  == 268 # Check that the number of columns is correct


################################################################################
# %% Z: Residualization data
# ==========================

# import importlib
# import ml_utils
# importlib.reload(ml_utils)
# from ml_utils import get_residualizer

if config['residualization']:
    from mulm.residualizer import Residualizer, ResidualizerEstimator

    residualization_columns = config['residualization']
    residualization_formula = "+".join(residualization_columns)
    residualizer = Residualizer(data=data, formula_res=residualization_formula)

    # Extract design matrix and pack it with X
    Z = residualizer.get_design_mat(data=data)
    residualizer_estimator = ResidualizerEstimator(residualizer)
        
    # Pack Z with X
    ZX = residualizer_estimator.pack(Z, X)
    Z_, X_ = residualizer_estimator.upack(ZX)
    assert np.all(X_ == X)
    assert Z.shape[1] + len(feature_columns) == ZX.shape[1]  # Check that the number of columns is correct

    # Finally, we will use ZX as X for the models, and Z as the residualization part
    X = ZX

################################################################################
# %% Run models using cross_validate (compute the recalls)
# ==============================================================

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score, roc_auc_score
from ml_utils import group_by_roi
roi_groups = group_by_roi(feature_columns)
# Convert roi_groups to indices
roi_groups = {roi:[feature_columns.index(x) for x in cols] for roi, cols in roi_groups.items()}


# Models
from ml_utils import make_models

models = make_models(n_jobs_grid_search=5, cv_val=cv_val,
                    residualization_formula=residualization_formula,
                    residualizer_estimator=residualizer_estimator,
                    roi_groups=roi_groups)

# Debug with onlys one model
if False:
    k = 'model-grpRoiLda+lrl2_resid-age+sex+site'
    models = {k:models[k]}

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
print(metrics_df)

metrics_historical_df = pd.DataFrame(rows_historical, columns=['model', 'test_balanced_accuracy', 
                                                               'test_roc_auc',
                                                               'test_recall_class_0', 'test_recall_class_1'])
print(metrics_historical_df)

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    metrics_df.to_excel(writer,            sheet_name="metrics")
    metrics_historical_df.to_excel(writer, sheet_name="metrics_historical", index=False)
print(f"\n✔  Saved {OUTPUT}")


################################################################################
# %% Explore individual ROI/feature
# =================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



def find_indices(feature_columns, items):
    return [i for i, x in enumerate(feature_columns) if x in items]
    
def make_selector_linear_model(features_set):
    indices = find_indices(feature_columns, features_set)
    selector = ColumnTransformer(
        [("select", "passthrough", indices)],
        remainder="drop",
    )
    return Pipeline([
        ("residualizer", residualizer_estimator),
        ("selector", selector),
        ("prep", StandardScaler()),
        ("clf", GridSearchCV(
            LogisticRegression(fit_intercept=False, class_weight="balanced"),
            {"C": 10. ** np.arange(-3, 1)},
            cv=cv_val, n_jobs=5, scoring="accuracy",
        )),
    ])

models_reduced = {
"Amygdala": make_selector_linear_model(
    ["Left Amygdala_GM_Vol", "Left Amygdala_CSF_Vol",
     "Right Amygdala_GM_Vol", "Right Amygdala_CSF_Vol"]),
"Hippocampus": make_selector_linear_model(
    ["Left Hippocampus_GM_Vol", "Left Hippocampus_CSF_Vol",
     "Right Hippocampus_GM_Vol", "Right Hippocampus_CSF_Vol"]),
"Middle Temporal Gyrus": make_selector_linear_model(
    ["Left Middle Temporal Gyrus_GM_Vol", "Left Middle Temporal Gyrus_CSF_Vol",
     "Right Middle Temporal Gyrus_GM_Vol", "Right Middle Temporal Gyrus_CSF_Vol"]),

"Left Amygdala": make_selector_linear_model(
    ["Left Amygdala_GM_Vol", "Left Amygdala_CSF_Vol"]),
"Right Amygdala": make_selector_linear_model(
    ["Right Amygdala_GM_Vol", "Right Amygdala_CSF_Vol"]),
"Left Hippocampus": make_selector_linear_model(
    ["Left Hippocampus_GM_Vol", "Left Hippocampus_CSF_Vol"]),
"Right Hippocampus": make_selector_linear_model(
    ["Right Hippocampus_GM_Vol", "Right Hippocampus_CSF_Vol"]),

"Left Amygdala GM": make_selector_linear_model(["Left Amygdala_GM_Vol"]),
"Right Amygdala GM": make_selector_linear_model(["Right Amygdala_GM_Vol"]),
"Left Hippocampus GM": make_selector_linear_model(["Left Hippocampus_GM_Vol"]),
"Right Hippocampus GM": make_selector_linear_model(["Right Hippocampus_GM_Vol"]),

"Left Amygdala+Hippocampus GM": make_selector_linear_model(
    ["Left Amygdala_GM_Vol", "Left Hippocampus_GM_Vol"]),

"Left Amygdala+Hippocampus GM+Middle Temporal Gyrus GM": make_selector_linear_model(
    ["Left Amygdala_GM_Vol", "Left Hippocampus_GM_Vol", "Left Middle Temporal Gyrus_GM_Vol", "Right Middle Temporal Gyrus_GM_Vol"]),
"Left Amygdala+Hippocampus GM+Middle Temporal Gyrus CSF": make_selector_linear_model(
    ["Left Amygdala_GM_Vol", "Left Hippocampus_GM_Vol", "Left Middle Temporal Gyrus_CSF_Vol", "Right Middle Temporal Gyrus_CSF_Vol"]),
}


rows = []

for mod, model in models_reduced.items():
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

metrics_df_single = pd.concat(rows)
print(metrics_df_single)

metrics_df = pd.concat([metrics_df, metrics_df_single], axis=0)
print(metrics_df)


with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    metrics_df.to_excel(writer,            sheet_name="metrics")
print(f"\n✔  Saved {OUTPUT}")
"""
                                    model  test_balanced_accuracy  test_roc_auc  test_recall_class_0  test_recall_class_1
0          model-lrl2cv_resid-age+sex+site                0.691230      0.683968             0.635238             0.747222
1        model-lrenetcv_resid-age+sex+site                0.522500      0.538148             0.786667             0.258333
2        model-svmrbfcv_resid-age+sex+site                0.580476      0.430979             0.727619             0.433333
3        model-forestcv_resid-age+sex+site                0.549524      0.617116             0.865714             0.233333
4            model-gbcv_resid-age+sex+site                0.445952      0.437963             0.541905             0.350000
5          model-mlp_cv_resid-age+sex+site                0.537500      0.524524             1.000000             0.075000
6  model-grpRoiLda+lrl2_resid-age+sex+site                0.703175      0.690185             0.661905             0.744444
"""
# %% 
# import importlib
# import utils.sklearn_utils
# importlib.reload(utils.sklearn_utils)
# from utils.sklearn_utils import classification_report_cv




