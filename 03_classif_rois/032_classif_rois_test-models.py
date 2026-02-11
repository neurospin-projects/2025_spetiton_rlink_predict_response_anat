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

from sklearn.model_selection import StratifiedKFold, LeaveOneOut


from config import config, cv_val

from ml_utils import create_print_log
from ml_utils import get_y, get_X, get_residualizer
from ml_utils import dict_cartesian_product, run_parallel, run_sequential
from ml_utils import get_linear_coefficients, pipeline_behead
from ml_utils import PredefinedSplit
from ml_utils import GroupFeatureTransformer
from ml_utils import fit_predict_binary_classif, ClassificationScorer

# Default output prefix
config['prefix'] = "032_classif_rois_test-models"

# Set output paths
# config['log_filename'] = config['prefix'] + ".log"
# config['cachedir'] = config['prefix'] + ".cachedir"


# Print log function
if 'log_filename' in config:
    print_log = create_print_log(config['log_filename'])
    print_log('###################################################################')
    print_log('## %s' % datetime.now().strftime("%Y-%m-%d %H:%M"))
    print_log(config)
else:
    print_log = print  # Fallback to print if no log file is specified

# Create cachedir
from joblib import Memory
if 'cachedir' in config:
    memory = Memory(config['cachedir'], verbose=0)


################################################################################
# %% 2. Read Data
# ===============

data = pd.read_csv(config['input_data'])

# I.1 Target variable => y
y = get_y(data, target_column=config['target'],
          remap_dict=config['target_remap'], print_log=print_log)

# I.2 X: Input Data
# Select Input = dataframe - (target + drop + residualization)
feature_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]

X = get_X(data, feature_columns, print_log=print_log)
X = X.values

# Multiply CSF columns by -1
csf_indices = np.array([i for i, col in enumerate(feature_columns) if 'CSF' in col])
X[:, (csf_indices)] *= -1

# I.3 Z: Residualization data

residualization_formula = False
Z_ncol = 0
if config['residualization']:  
    X, residualizer_estimator, residualization_formula = \
        get_residualizer(data, X, residualization_columns=config['residualization'],
                    print_log=print_log)
    Z, X_ = residualizer_estimator.upack(X)
    Z_ncol = Z.shape[1]

assert Z_ncol + len(feature_columns) == X.shape[1]  # Check that the number of columns is correct

# Check that X == [Z + X[feature_columns]]
X_ = get_X(data, feature_columns, print_log=print).values
X_[:, (csf_indices)] *= -1
assert np.all(X == residualizer_estimator.pack(Z, X_))



################################################################################
# %% Test many models using cross_validate (compute the recalls)
# ==============================================================

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score, roc_auc_score
from ml_utils import group_by_roi
roi_groups = group_by_roi(feature_columns)
roi_groups = {roi:[feature_columns.index(x) for x in cols] for roi, cols in roi_groups.items()}


# Models

from classification_models import make_models
models = make_models(n_jobs_grid_search=5, cv_val=cv_val,
                        residualization_formula=residualization_formula,
                        residualizer_estimator=residualizer_estimator,
                        roi_groups=roi_groups)


# Load the CV test split

cv_test = PredefinedSplit(json_file=config['cv_test'])


# Scorers including recall for each class

recall_scorers = {
    f"recall_class_{label}": make_scorer(recall_score, labels=[label], average="macro")
    for label in np.unique(y)
}

scorers = {
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "roc_auc": make_scorer(roc_auc_score),
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
scores = list()

for mod, model in models.items():
    print(f"Evaluating model: {mod}")
    cv_res = cross_validate(
        estimator=model, X=X, y=y,
        cv=cv_test, scoring=scorers,
        return_train_score=False, n_jobs=5, verbose=50)

    # Print average metrics
    scores.append([mod] + average_metrics(cv_res, metrics))

scores_df = pd.DataFrame(scores, columns=['model'] + metrics)

print(scores_df)

"""
                                     model  test_balanced_accuracy  test_roc_auc  test_recall_class_0  test_recall_class_1
0          model-lrl2cv_resid-age+sex+site                0.691230      0.691230             0.635238             0.747222
1        model-lrenetcv_resid-age+sex+site                0.531111      0.531111             0.906667             0.155556
2        model-svmrbfcv_resid-age+sex+site                0.612500      0.612500             0.700000             0.525000
3        model-forestcv_resid-age+sex+site                0.549524      0.549524             0.865714             0.233333
4            model-gbcv_resid-age+sex+site                0.445952      0.445952             0.541905             0.350000
5  model-grpRoiLda+lrl2_resid-age+sex+site                0.703175      0.703175             0.661905             0.744444
"""



# %%
