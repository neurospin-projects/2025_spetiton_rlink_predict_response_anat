'''
Supervised Classification: Group ROIs with LDA followed L2LR including Randomization
====================================================================================

'''
################################################################################
# %% 1. Initialization 
# ====================
import os.path
from datetime import datetime
import numpy as np
import pandas as pd

from config import config, cv_val

from ml_utils import create_print_log
from ml_utils import get_y, get_X, get_residualizer
from ml_utils import dict_cartesian_product, run_parallel, run_sequential
from ml_utils import get_linear_coefficients, pipeline_behead
from ml_utils import PredefinedSplit
from ml_utils import GroupFeatureTransformer
from ml_utils import fit_predict_binary_classif, ClassificationScorer
#from ml_utils import multipletests

from classification_models import make_models

# Default output prefix
config['prefix'] = "033_classif_rois_grpRoiLdaLrL2-randomize"
config['input_permutations'] = config['prefix'] + "_permutations_seeds.csv"
 

# Set output paths
config['output_rois'] = \
    os.path.join(config['output_models'], config['prefix'] + "_rois.csv")
config['output_predictions_scores_feature-importance'] = \
    os.path.join(config['output_models'], config['prefix'] + "_predictions-scores_feature-importance.xlsx")
#config['log_filename'] = config['prefix'] + ".log"
#config['cachedir'] = config['prefix'] + ".cachedir"


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
# %% 3. Grouping features by ROIs (utils)
# =======================================
#
# Colinearity is high between features (left, right, CSF, GM) of the same ROI,
# to measure the importance of each ROI. We will group the features by ROI after
# preprocessing and before prediction head and compute the feature importance for each ROI.
# 1. Group input columns by ROI name

from ml_utils import group_by_roi
roi_groups = group_by_roi(feature_columns)
roi_groups = {roi:[feature_columns.index(x) for x in cols] for roi, cols in roi_groups.items()}

pd.DataFrame([[roi, len(subrois), ",".join(subrois)] for roi, subrois in group_by_roi(feature_columns).items()],
            columns=['ROI', 'N_features', 'Features']).to_csv(config['output_rois'], index=False)

assert len(feature_columns) == 268
assert len(roi_groups) == 71
mask_gm = np.array(['GM' in s for s in feature_columns])
mask_csf = np.array(['CSF' in s for s in feature_columns])
mask_left = np.array(['Left' in s for s in feature_columns])
mask_right = np.array(['Right' in s for s in feature_columns])
# Check the number of features
assert np.sum(mask_gm) == 134
assert np.sum(mask_csf) == 135
assert np.sum(mask_left) == 126
assert np.sum(mask_right) == 126
# Check the number of features per ROI

np.sum(['GM' in s for s in feature_columns]) == 134
np.sum(['CSF' in s for s in feature_columns]) == 135
np.sum(['Left' in s for s in feature_columns]) == 126
np.sum(['Right' in s for s in feature_columns]) == 126


################################################################################
# %% 4. Configure models, CV and permutation scheme
# ==================================================

# Models

from classification_models import make_models
models = make_models(n_jobs_grid_search=5, cv_val=cv_val,
                        residualization_formula=residualization_formula,
                        residualizer_estimator=residualizer_estimator,
                        roi_groups=roi_groups)

model_names = ['model-grpRoiLda+lrl2_resid-age+sex+site', 'model-lrl2cv_resid-age+sex+site']
models = {mod: model for mod, model in models.items() if mod in model_names}


# Permutation/CV scheme

# Permutation 0 is without permutations
def permutation(x, random_state=None):
    if random_state == 0:
        return(x)
    if random_state is not None:
        np.random.seed(seed=random_state)
    return np.random.permutation(x)


permutation_seed =  pd.read_csv(config['input_permutations']).perm.values
#permutation_seed = [0]
nperms = len(permutation_seed) - 1
# df = pd.read_excel("/home/ed203246/git/2025_spetiton_rlink_predict_response_anat/03_classif_rois/models/20_ml-classifLiResp_predictions_scores_feature-importance_v-20250708.xlsx", sheet_name='predictions')
# perms = pd.DataFrame(dict(perm=[int(perm.split('-')[1]) for perm in df.perm.unique()]))
# perms.to_csv(config['input_permutations'])

# Load the CV test split
cv_test = PredefinedSplit(json_file=config['cv_test'])

# {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i'  % fold):
    (X, permutation(y, perm), train_index, test_index)
    for perm in permutation_seed
    for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
print(cv_test_dict_Xy.keys())

# => Output:
models_cv = dict_cartesian_product(models, cv_test_dict_Xy)

################################################################################
# %% 5. Fit models
# ================

# 
# fit_predict = memory.cache(fit_predict_binary_classif)
fit_predict = fit_predict_binary_classif
res_cv = run_parallel(fit_predict, models_cv, verbose=50)
# res_cv = run_sequential(fit_predict, models_cv, verbose=50)
# {k:fit_predict(*v, verbose=50) for k, v in models_cv.items()}
#res_cv[[k for k in res_cv.keys()][0]].keys()

################################################################################
# %% 6. Classifications metrics
# =============================

reducer = ClassificationScorer()
predictions_df = reducer.predictions_dict_toframe(res_cv)
predictions_metrics_df = reducer.prediction_metrics(predictions_df)
print(predictions_metrics_df)
"""
                                        model      perm  balanced_accuracy   roc_auc
0        model-grpRoiLda+lrl2_resid-age+sex+site  perm-000           0.703175  0.690185
1  model-grpRoiLdaClust2+lrl2_resid-age+sex+site  perm-000           0.659087  0.691481
2  model-grpRoiLdaClust3+lrl2_resid-age+sex+site  perm-000           0.659087  0.693968
3                  model-lrl2_resid-age+sex+site  perm-000           0.691230  0.683968
4                   model-mlp_resid-age+sex+site  perm-000           0.502778  0.515635
"""

predictions_metrics_pvalues_df = reducer.prediction_metrics_pvalues(predictions_metrics_df)
print(predictions_metrics_pvalues_df)

""""
                                     model      perm  balanced_accuracy   roc_auc  balanced_accuracy_h0_pval  roc_auc_h0_pval  balanced_accuracy_h0_mean  roc_auc_h0_mean  balanced_accuracy_h0_std  roc_auc_h0_std
0  model-grpRoiLda+lrl2_resid-age+sex+site  perm-000           0.703175  0.690185                      0.001            0.007                   0.498896         0.498644                  0.059656        0.076297
1          model-lrl2cv_resid-age+sex+site  perm-000           0.691230  0.683968                      0.000            0.001                   0.498254         0.499443                  0.059400        0.073973
"""

################################################################################
# %% 7. Feature importance
# ========================

from ml_utils import dict_to_frame
from ml_utils import mean_sd_tval_pval_ci
#from statsmodels.stats.multitest import multipletests


# Provide feature names for each model
models_features_names = {mod: feature_columns for mod in models.keys()}
models_features_names['model-grpRoiLda+lrl2_resid-age+sex+site'] = roi_groups.keys()
#features_names['model-grpRoiLdaClust2+lrl2_resid-age+sex+site'] = ["clust-%02i" % i for i in range(2)]
#features_names['model-grpRoiLdaClust3+lrl2_resid-age+sex+site'] = ["clust-%02i" % i for i in range(3)]

from ml_utils import stack_features_dicts, features_statistics, features_statistics_pvalues


features_df = stack_features_dicts(res_cv,
                models_features_names=models_features_names,
                importances=['coefs', 'forwd', 'feature_auc'])

features_stats = features_statistics(features_df)
features_stats_pvals = features_statistics_pvalues(features_stats)


################################################################################
# %% 8. Save results
# ==================

with pd.ExcelWriter(config['output_predictions_scores_feature-importance']) as writer:#, mode="a", if_sheet_exists="replace") as writer:
    predictions_metrics_pvalues_df.to_excel(writer, sheet_name='predictions_metrics_pvalues', index=False)
    predictions_df.to_excel(writer, sheet_name='predictions', index=False)
    for (mod, stat), df in features_stats_pvals.items():
        sheet_name = '__'.join([mod, stat])
        # print(sheet_name)
        df.to_excel(writer, sheet_name=sheet_name, index=False)


