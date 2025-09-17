################################################################################
# Imports
# -------

# %%

import pandas as pd, numpy as np
import sys, os, os.path, tempfile, time, logging, json
import matplotlib.pyplot as plt, seaborn as sns # plotting

# univariate statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Dataset
from sklearn.datasets import make_classification

from itertools import product

# Joblib
from joblib import Parallel, delayed
from joblib import Memory
from joblib import cpu_count

# Metrics
import sklearn.metrics as metrics

# Models
from sklearn.ensemble import StackingClassifier

# Resampling
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
# Set pandas display options
pd.set_option('display.max_colwidth', None)  # No maximum width for columns
pd.set_option('display.width', 1000)  # Set the total width to a large number

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"

# outputs
OUTPUT_DIR = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/reports/classification_results/"
# %%
config_file = ROOT+'models/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4/supervised_classification_config.json'

with open(config_file) as json_file:
    config = json.load(json_file)

sys.path.append(config["nitk_path"])

# Utils
import nitk
from nitk.sys_utils import import_module_from_path
from nitk.pandas_utils.dataframe_utils import expand_key_value_column
from nitk.pandas_utils.dataframe_utils import describe_categorical
from nitk.ml_utils.dataloader_table import get_y, get_X
from nitk.ml_utils.residualization import get_residualizer
from nitk.ml_utils.cross_validation import PredefinedSplit

################################################################################
# Read config file (Parameters)
# Import models
sys.path.append(os.path.dirname(config["models_path"]))
root, _ = os.path.splitext(os.path.basename(config["models_path"]))

config['log_filename'] = os.path.splitext(config_file)[0] + ".log"
config['output_filename'] = os.path.splitext(config_file)[0] + "_scores_no_balanced_weights.csv"
config['stratification_sse'] = os.path.splitext(config_file)[0] + "_stratification-sse.csv"
config['cv5test'] = os.path.splitext(config_file)[0] + "_cv5test.json"

config['cachedir'] = os.path.splitext(config_file)[0] + "/cachedir"

# Import the module
import_module_from_path(config["models_path"])
from classification_models import make_models

def print_log(*args):
    with open(config['log_filename'], "a") as f:
        print(*args, file=f)


print_log('###########################################################')
print_log(config)

################################################################################
# Read Data
# %%
data = pd.read_csv(config['input_filename'])
print(data)
################################################################################
# Target variable => y
# %%
y = get_y(data, target_column=config['target'],
          remap_dict=config['target_remap'], print_log=print_log)

################################################################################
# X: Input Data
# Select Input = dataframe - (target + drop + residualization)
# %%

input_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]

X = get_X(data, input_columns, print_log=print_log)
X = X.values
print(len(input_columns))
# %%
################################################################################

################################################################################
# Fit/Predict (Cached) Function for Parallel Execution
# ----------------------------------------------------
#
# Principles:
#
# 1. Define dictionary of models including gridseach model selection, where parallelization
#    is controlled by a `n_jobs_grid_search` parameter.
#
# 2. Define a fit/predict (Cached) function ``fit_predict()`` for parallel execution.
#    It returns everything required for further analysis, fitted models, test prediction, test indices, etc.
#
# 3. Run  ``fit_predict()`` using a sequential ``run_sequential()``) or parallel ``run_parallel()`` execution:
#
#    - Execution follows a "flat" map-reduce scheme  of ``fit_predict()`` function (the mapper) over over all
#      combinations of models * CV-folds.
#    - ``run_xx()`` return everything needed for further analysis, such as fitted model, test prediction, test indices, etc.
#    - ``run_xx()`` also returns predictions into a DataFrame for further computation of metrics.
#    - Predictions DataFrame over models * CV-folds can be save into a file.
#    - Parallelization can be controlled at every steps:
#      (i) inner loop of model selection using ``make_models(n_jobs_grid_search)``
#      (ii) outer loop of model x cv execution. Prediction are save into a csv file.
#
#  4. The computation of classification metrics is done afterwards, using dataFrame.




################################################################################
# 2. Define Fit/Predict (Cached) function for Parallel Execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# 2.1 ``fit_predict``: performs on demand recomputing using the
#     `Joblib Memory class <https://joblib.readthedocs.io/en/latest/memory.html>`_.
#     Returns ``result_dict`` with fitted model, test predictions, and test indices.


memory = Memory(config['cachedir'], verbose=0)


# @memory.cache
def fit_predict_binary_classif(estimator_name, estimator, fold, X, y,
                               train_idx, test_idx,
                               verbose=0):
    """Fit and predict using on demand recomputing Joblib Memory class. 

    Parameters
    ----------
    estimator_name : str
        _description_
    estimator : Sklearn fitted estimator
        _description_
    fold : int/str
        _description_
    X : Array
        _description_
    y : Array
        _description_
    train_idx : _type_
        _description_
    test_idx : _type_
        _description_
    verbose : int
        verbose if >= 10
    Returns
    -------
    _type_
        _description_
    """
    if verbose >= 10:
        print("Fit %s" % estimator_name)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    estimator = clone(estimator).fit(X_train, y_train)

    # Predicted labels
    pred_labs_test = estimator.predict(X_test)
    pred_labs_train = estimator.predict(X_train)

    # Predicted values before threshold
    

    try:
        pred_proba_test = estimator.predict_proba(X_test)[:, 1]
        pred_proba_train = estimator.predict_proba(X_train)[:, 1]
    except AttributeError:
        pred_proba_test = pred_proba_train = None
    try:
        pred_decision_function_test = estimator.decision_function(X_test)
        pred_decision_function_train = estimator.decision_function(X_train)
    except AttributeError:
        pred_decision_function_test = pred_decision_function_train = None

    result_dict = dict(
        # Keys:
        model=estimator_name, fold=fold, test_idx=test_idx,
        # Estimator:
        estimator=estimator,
        # Test predictions:
        pred_labs_test=pred_labs_test,
        pred_proba_test=pred_proba_test,
        pred_decision_function_test=pred_decision_function_test,
        # Train predictions:
        pred_labs_train=pred_labs_train,
        pred_proba_train=pred_proba_train,
        pred_decision_function_train=pred_decision_function_train,
        train_idx=train_idx)

    return result_dict

# %%
# 2.2 ``predictions_dataframe(result_dict)``: Transform prediction result into DataFrame.


def predictions_dataframe(result_dict):
    """Transform prediction result into DataFrame.

    Parameters
    ----------
    result_dict : dict
        Prediction result from fit_predict()

    Returns
    -------
    DataFrame
        _description_
    """
    cols = ['model', 'fold',
            'pred_labs_test', 'pred_proba_test', 'pred_decision_function_test',
            'test_idx']
    return pd.DataFrame({col: result_dict[col] for col in cols})

# %%
# (Flat) iterator over models and cv

def models_cv_iterator(models, cv):
    if not isinstance(cv, dict):
        cv = {fold_num: (train_idx, test_idx) for
              fold_num, (train_idx, test_idx) in enumerate(cv.split(X, y))}
    for (estimator_name, estimator), (fold_num, (train_idx, test_idx)) in\
            product(models.items(), cv.items()):
        yield estimator_name, estimator, fold_num, train_idx, test_idx
        
################################################################################
# 3. Run Fit/Predict
# ~~~~~~~~~~~~~~~~~~
#

# %%
# **Sequential execution function**
#

def run_sequential(fit_predict, models, cv_test, memory=None, clear_cache=False,
                   verbose=0):

    # Clear cache
    if memory and clear_cache:
        memory.clear()

    # fit_predict = fit_predict_binary_classif
    # for estimator_name, estimator, fold_num, train_idx, test_idx in models_cv_iterator(models, cv_test):
    #     pass

    start_time = time.time()
    fitted_models_cv_list = [fit_predict(estimator_name, estimator,
                                         fold_num, X, y, train_idx, test_idx,
                                         verbose=verbose)
                             for estimator_name, estimator, fold_num, train_idx, test_idx
                             in models_cv_iterator(models, cv_test)]
    if verbose > 0:
        print('Sequential execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))

    predictions_df = pd.concat([predictions_dataframe(r)
                               for r in fitted_models_cv_list])

    return fitted_models_cv_list, predictions_df


# %%
# **Parallel execution function**


def run_parallel(fit_predict, models, cv_test, memory=None, clear_cache=False,
                 verbose=0):
    # Clear cache
    if memory and clear_cache:
        memory.clear()

    parallel = Parallel(n_jobs=cpu_count(only_physical_cores=True))

    start_time = time.time()
    # here we need to call delayed to make sure the fit_predict function runs in parallel
    # otherwise it would be running sequentially
    fitted_models_cv_list = \
        parallel(delayed(fit_predict)(estimator_name, estimator,
                                      fold_num, X, y, train_idx, test_idx,
                                      verbose=verbose)
                 for estimator_name, estimator, fold_num, train_idx, test_idx
                 in models_cv_iterator(models, cv_test))
    if verbose > 0:
        print('Parallel execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))

    predictions_df = pd.concat([predictions_dataframe(r)
                               for r in fitted_models_cv_list])

    return fitted_models_cv_list, predictions_df



################################################################################
# 4. Recompute scores from saved predictions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
def classification_metrics(predictions_df, groupby=['model', 'fold']):
    models_cv_classif_scores = \
        [list(keys) +
         [
            metrics.balanced_accuracy_score(
                y[df["test_idx"]], df["pred_labs_test"]),
            # take decision_function when exists else proba
            metrics.roc_auc_score(
                y[df["test_idx"]],
                np.where(~np.isnan(df['pred_decision_function_test']),
                         df['pred_decision_function_test'], df['pred_proba_test'])
            )
        ] for keys, df in predictions_df.groupby(groupby)
        ]

    scores_colnames = ['balanced_accuracy', 'roc_auc']
    models_cv_classif_scores = \
        pd.DataFrame(models_cv_classif_scores,
                     columns=groupby + scores_colnames)

    return models_cv_classif_scores

def cv_stratification_metric(df, factors, cv):
    """Metrics (SSE) that evaluate the quality kfolds stratification for many
    factors. Sum accross fold SSE (true proportions (from df[factors], 
    fold proportions (from df[test, factors])

    Parameters
    ----------
    df : _type_
        _description_
    factors : _type_
        _description_
    cv : _type_
        _description_
    """
    def proportions_byfactors(df, factors):
        counts = df.groupby(factors).size()
        prop = counts / counts.sum()
        return prop

    prop_tot = proportions_byfactors(df=df, factors=factors)

    sse = 0
    for train_index, test_index in cv.split(X, y):
        prop_fold = proportions_byfactors(df=df.iloc[test_index], factors=factors)
        #print(np.sum((prop_tot - prop_fold) ** 2))
        sse += np.sum((prop_tot - prop_fold) ** 2)

    return sse

# %%
# Run Everything
# ~~~~~~~~~~~~~~

# 1. Make models without inner loop parallelization
n_splits_val = 5
cv_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=42)
if config['residualization']:  
    X, residualizer_estimator, residualization_formula = \
        get_residualizer(data, X, residualization_columns=config['residualization'],
                    print_log=print_log)
print(residualization_formula) 
# %%
class_weight= None #"balanced"
models = make_models(n_jobs_grid_search=1, cv_val=cv_val,
                     residualization_formula=residualization_formula,
                     residualizer_estimator=residualizer_estimator, class_weight=class_weight)

df = pd.DataFrame(dict(y=y, site=data['site']))

cv_test = PredefinedSplit(json_file=config['cv5test'])
print(cv_stratification_metric(df, factors=['y', 'site'], cv=cv_test))  # should be 0.080669

cv_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=8) # change random_state to seed_with_min_sse
print(cv_stratification_metric(df, factors=['y', 'site'], cv=cv_test)) # should be 0.080669 if random_state=4, and 0.07846329621661646 if random_state=8
PredefinedSplit([(train, test) for train, test in cv_test.split(X, y)]).to_json(\
    "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/models/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4/supervised_classification_config_cv5test_seed8.json")
quit()

if class_weight=="balanced":
    predictions_file = os.path.join(OUTPUT_DIR, "predictions_balanced_weights.csv")
else : predictions_file = os.path.join(OUTPUT_DIR, "predictions.csv")

if os.path.exists(predictions_file):
    # 2.& 3 Run fit/predict
    fitted_models_cv_list, predictions_df = \
        run_sequential(fit_predict_binary_classif, models, cv_test, memory,
                       clear_cache=False, verbose=1)

    # fitted_models_cv_list, predictions_df = \
    #     run_parallel(fit_predict_binary_classif, models, cv_test, memory,
    #                 clear_cache=True, verbose=1)

    # Re-run using cached computation
    # fitted_models_cv_list, predictions_df = \
    #     run_parallel(fit_predict_binary_classif, models, cv_test, memory,
    #                  clear_cache=False, verbose=1)

    # Save predictions in csv file  with ``estimator_name``, ``fold_num``, and ``test_idx`` as primary keys
    predictions_df.to_csv(predictions_file, index=False)

predictions_df=pd.read_csv(predictions_file)

# 4. Compute classification results
models_cv_classif_scores = classification_metrics(predictions_df,
                                                  groupby=['model', 'fold'])

print(models_cv_classif_scores.drop(
    columns=['fold']).groupby(['model']).mean())

"""

                                   balanced_accuracy   roc_auc
model                                                         
model-forestcv_resid-age+sex+site           0.590992  0.646296
model-gbcv_resid-age+sex+site               0.472540  0.459286
model-lrenetcv_resid-age+sex+site           0.560317  0.616376
model-lrl2cv_resid-age+sex+site             0.526151  0.605608
model-svmrbfcv_resid-age+sex+site           0.562540  0.594947

for seed = 4 and class_weight = None

"""
quit()

################################################################################
# 5. Add new model - stacked classifier
# ~~~~~~~~~~~~~~~~

# %%
estimators_ = [(model_name, model) for model_name, model in models.items()]
stacked_clf = StackingClassifier(
    estimators_, final_estimator=lm.LogisticRegression())
models["stacked_clf"] = stacked_clf

fitted_models_cv_list, predictions_df = \
    run_parallel(fit_predict_binary_classif, models, cv_test, memory,
                 clear_cache=False, verbose=10)

models_cv_classif_scores = classification_metrics(predictions_df,
                                                  groupby=['model', 'fold'])
print(models_cv_classif_scores.drop(
    columns=['fold']).groupby(['model']).mean())


################################################################################
# 5. Learning curve
# ~~~~~~~~~~~~~~~~~

# %%

# Create CV splitter using 10%, 33%, 55%, 78%, 100% of the training data
train_sizes = np.array([0.1, 0.33, 0.55, 0.78, 1.])

def sub_sample(idx, size):
    return idx[:round(len(idx) * size)]

# CV spliter key is made with: "fold-%i_size-%.2f"
cv_test_lc = {"fold-%i_size-%.2f" % (fold_num, size): (sub_sample(train_idx, size), test_idx) for
              fold_num, (train_idx, test_idx) in enumerate(cv_test.split(X, y)) for size in train_sizes}

print("CV splitter using 10%, 33%, 55%, 78%, 100% of the training data")
print(cv_test_lc.keys())

# Fit the models over all folds x sample sizes
fitted_models_cv_lc_list, predictions_df = \
    run_parallel(fit_predict_binary_classif, models, cv_test_lc, memory,
                 clear_cache=False, verbose=10)

# Expands folds column into two columns 'fold' and "size"
predictions_df = expand_key_value_column(df=predictions_df, col='fold')

# Compute classification metrics for each 'model', 'fold', and "size":
models_cv_classif_scores = classification_metrics(predictions_df,
                                                  groupby=['model', 'fold', "size"])


models_size_classif_score = models_cv_classif_scores.drop(
    columns=['fold']).groupby(['model', "size"]).mean().reset_index()

print(models_size_classif_score)

sns.lineplot(data=models_size_classif_score, x="size", y="roc_auc", hue='model')


# %%


################################################################################
# Fit/Predict and Compute Test Score (CV) with Cached Computation
# ---------------------------------------------------------------

# %%

memory = Memory(cachedir, verbose=0)
cross_validate_cached = memory.cache(cross_validate)

fitted_models_cv_scores = dict()
start_time_total = time.time()
for name, model in models.items():
    # name, model = "lrl2_cv", models["lrl2_cv"]
    start_time = time.time()
    fitted_model_cv_scores_ = cross_validate_cached(estimator=model, X=X, y=y, cv=cv_test,
                                                    n_jobs=n_splits_test,
                                                    scoring=metrics_names,
                                                    return_estimator=True,
                                                    return_indices=True)
    print(name, 'Elapsed time: \t%.3f sec' % (time.time() - start_time))
    fitted_models_cv_scores[name] = fitted_model_cv_scores_

print('Total Elapsed time: \t%.3f sec' % (time.time() - start_time_total))


################################################################################
# Average Test Scores (CV) and save it to a file
# ----------------------------------------------

# %%
test_stat = [[name] + [res["test_" + metric].mean() for metric in metrics_names]
             for name, res in fitted_models_cv_scores.items()]

test_stat = pd.DataFrame(test_stat, columns=["model"]+metrics_names)
test_stat.to_csv(os.path.join(OUTPUT_DIR, "test_stat.csv"))
print(test_stat)
