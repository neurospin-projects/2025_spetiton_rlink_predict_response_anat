'''
Supervised Classification
=========================

rsync -azvu --exclude study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4_task-predLiResp/20_ml-classifLiResp.cachedir study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4_task-predLiResp triscotte.intra.cea.fr:/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/edouard/

'''

# %% 1. Initialization 
# ====================

# # %% Imports
# # ----------

# # System
# import sys
# import os
# import os.path
# import time
# import json
# from datetime import datetime


# # Scientific python
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats
# from statsmodels.stats.multitest import multipletests

# # Univariate statistics
# # import statsmodels.api as sm
# # import statsmodels.formula.api as smf
# # import statsmodels.stats.api as sms

# # from itertools import product

# # Models
# from sklearn.base import clone
# # from sklearn.decomposition import PCA
# # import sklearn.linear_model as lm
# # import sklearn.svm as svm
# # from sklearn.neural_network import MLPClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.ensemble import GradientBoostingClassifier
# # from sklearn.ensemble import BaggingClassifier
# # from sklearn.ensemble import StackingClassifier

# # Metrics
# import sklearn.metrics as metrics

# # Resampling
# # from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_validate
# # from sklearn.model_selection import cross_val_predict
# # from sklearn.model_selection import train_test_split
# # from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold, LeaveOneOut
# #from sklearn import preprocessing

# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import make_pipeline
# from sklearn import preprocessing
# import sklearn.linear_model as lm
# from sklearn.compose import ColumnTransformer

# # Set pandas display options
# pd.set_option('display.max_colwidth', None)  # No maximum width for columns
# pd.set_option('display.width', 1000)  # Set the total width to a large number

from init import *

# Default output prefix
config['prefix'] = "rlink_classiLiResp_anatROI"

# Set output paths
config['log_filename'] = config['prefix'] + ".log"
config['cachedir'] = config['prefix'] + ".cachedir"

# Print log function
print_log = create_print_log(config['log_filename'])
print_log('###################################################################')
print_log('## %s' % datetime.now().strftime("%Y-%m-%d %H:%M"))
print_log(config)

################################################################################
# %% Read config file 
# --------------------

# #config_file = '/home/ed203246/git/nitk/scripts/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4_task-predLiResp/20_ml-classifLiResp_config.json'
# config_file = '/home/ed203246/git/2025_spetiton_rlink_predict_response_anat/03_classif_rois/20_ml-classifLiResp_config.json'
# with open(config_file) as json_file:
#      config = json.load(json_file)

# # LD_LIBRARY_PATH
# if "LD_LIBRARY_PATH" in config:
#     for path in config["LD_LIBRARY_PATH"]:
#         sys.path.append(path) # Add to system path

#from nitk.ml_utils.config import initialize_config
# config = initialize_config(config_file, config=config)

from joblib import Memory
memory = Memory(config['cachedir'], verbose=0)

################################################################################
# %% Set Output files
# -------------------

# Comment  the next line to avoid overwriting existing files
# config['output_repeatedcv'] = os.path.join(config['output_models'], config['prefix'] + "_repeatedcv.xlsx")
# config['output_stratification'] = os.path.join(config['output_models'], config['prefix'] + "_stratification-sse.csv")
# config['output_cv_test'] = config['prefix'] + "_cv-5cv.json"
config['input_cv_test'] = config['prefix'] + "_cv-5cv.json"
config['input_permutations'] = config['prefix'] + "_permutations.csv"

# config['output_predictions_scores_feature-importance'] = \
#     os.path.join(config['output_models'], config['prefix'] + "_predictions_scores_feature-importance.xlsx")

config['output_RoiGrdLda_coefs']= \
    os.path.join(config['output_models'], config['prefix'] + "_RoiGrdLda_coefs.xlsx")

config['output_feature_correlation_matrix']= \
    os.path.join(config['output_reports'], config['prefix'] + "_features_corr_matrix")
    
# n_splits_val = 5
# cv_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=42)

# ################################################################################
# # %% Additional imports (utils)
# # -----------------------------

# import nitk
# from nitk.sys_utils import import_module_from_path, create_print_log
# # from nitk.pandas_utils.dataframe_utils import expand_key_value_column
# # #from nitk.pandas_utils.dataframe_utils import describe_categorical
# # from nitk.ml_utils.dataloader_table import get_y, get_X
# # from nitk.ml_utils.residualization import get_residualizer
# # from nitk.jobs_utils import run_sequential, run_parallel
# # from nitk.ml_utils.cross_validation import PredefinedSplit
# # from nitk.python_utils import dict_cartesian_product
# # from nitk.ml_utils.sklearn_utils import pipeline_behead, pipeline_split, get_linear_coefficients
# # from nitk.ml_utils.custom_models import GroupFeatureTransformer

# # from nitk.ml_utils.iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# # Import models
# classification_models = import_module_from_path(config["models_path"])
# print_log = create_print_log(config)

print_log('###################################################################')
print_log('## %s' % datetime.now().strftime("%Y-%m-%d %H:%M"))
print_log(config)


################################################################################
# %% Read Data
# ------------

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
# %% 2. CV Stratification scheme
# ==============================
#
# make_splits() function 
# Find a 5CV split stratified for target and site
# Sample distribution (proportions for each response and site) in the original data
# Fold distribution (proportions for each response and site) in the original data
# Compute distribution error (Fold - sample)
# Sum arross folds the squared error (SSE) to evaluate the quality of the stratification

def cv_stratification_metric(df, factors, cv):
    """Metrics (SSE) that evaluate the quality kfolds stratification for many
    factors. Sum across fold SSE (true proportions (from df[factors], 
    fold proportions (from df[test_idx, factors])

    Parameters
    ----------
    df : _type_
        _description_
    factors : _type_
        _description_
    cv : _type_
        _description_
    """
    def make_cartesian_series_from_factors(df, factors, value=0):
        levels = [df[f].unique() for f in factors]
        index = pd.MultiIndex.from_product(levels, names=factors)
        return pd.Series(value, index=index)

    def proportions_byfactors(df, factors):
        counts = df.groupby(factors).size()
        prop = counts / counts.sum()
        return prop
    
    empty = make_cartesian_series_from_factors(df, factors, value=0)
    size_exp = np.prod([len(df[f].unique()) for f in factors])
    assert empty.shape[0] == size_exp

    # Target proportion
    prop_tot = (proportions_byfactors(df=df, factors=factors) + empty).fillna(0)

    count_tot = (df.groupby(factors).size() + empty).fillna(0)
    weights = count_tot / (count_tot ** 2).sum()
    assert np.allclose((count_tot * weights).sum(), 1)

    # Compute the sum of squared error (SSE) for each fold
    # SSE = sum((fold_proportions - total_proportions) * weights)
    sse, sse_weighted = 0, 0
    for train_index, test_index in cv.split(X, y):
        prop_fold = proportions_byfactors(df=df.iloc[test_index], factors=factors)
        sse += np.sum(((prop_fold - prop_tot)) ** 2)
        sse_weighted += np.sum(((prop_fold - prop_tot) * weights) ** 2)
        
    return sse, sse_weighted


def make_splits(X, y, groups_df, factors, n_splits=5):
    """Make splits for repeated CV. Measure the quality of the CV stratification
    using the sum of squared error (SSE) for each fold. 
    
    """
    n_splits_test = 5
    cv_test = StratifiedKFold(n_splits=n_splits_test,
                            shuffle=True, random_state=42)


    # github https://github.com/trent-b/iterative-stratification
    # paper https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10

    all_idx = np.arange(len(groups_df))
    cv_id = PredefinedSplit([[all_idx, all_idx] for i in range(5)])
    sse_, sse_weighted_ = cv_stratification_metric(groups_df, factors=factors, cv=cv_id)
    assert sse_ == sse_weighted_ == 0

    sse, sse_weighted = [], []
    rcvs = dict()
    for seed in range(100):
        #seed = 4
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits_index_mskf = [(train_index, test_index) for train_index, test_index in
                            mskf.split(X, groups_df[factors])]
        mskf = PredefinedSplit(splits_index_mskf)
        rcvs["mskf5cv-%i" % seed] = mskf
        sse_, sse_weighted_ = cv_stratification_metric(groups_df, factors=factors, cv=mskf)
        sse.append(["mskf5cv", seed, sse_, sse_weighted_])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        #splits_index_skf = [test_index for train_index, test_index in skf.split(X, y)]
        rcvs["skf5cv-%i" % seed] = skf
        sse_, sse_weighted_ = cv_stratification_metric(groups_df, factors=factors, cv=skf)
        sse.append(["skf5cv", seed, sse_, sse_weighted_])

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        #splits_index_skf = [test_index for train_index, test_index in skf.split(X, y)]
        rcvs["skf10cv-%i" % seed] = skf
        sse_, sse_weighted_ = cv_stratification_metric(groups_df, factors=factors, cv=skf)
        sse.append(["skf10cv", seed, sse_, sse_weighted_])


    rcvs['loo-all'] = LeaveOneOut()

    # Save stratification SSE
    sse = pd.DataFrame(sse, columns=['method', 'seed', 'sse', 'sse_weighted'])
    sse.sort_values(by='sse', inplace=True)

    return rcvs, sse


################################################################################
# %% Repeated CV using cross_validate
# ===================================
# 
# - Requires a make_splits() function
# - Use `cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#>`_`
# Choose `scoring functions <https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers>`_ 

groups_df = pd.DataFrame(dict(y=y, site=data['site']))
factors = ['y', 'site']
    
if "output_repeatedcv" in config:
    
    # 1. Repeated CV

    rcvs, sse = make_splits(X, y, groups_df, factors, n_splits=5)

    if 'output_cv_stratification' in config:
        sse.to_csv(config['output_stratification'], index=False)


    # 2.  Models (Do not parallelize grid search)
    models = classification_models.make_models(n_jobs_grid_search=1, cv_val=cv_val,
                        residualization_formula=residualization_formula,
                        residualizer_estimator=residualizer_estimator)

    print("Models (N=%i)" % len(models), models.keys())
    metrics_names = ['test_%s' % m for m in config["metrics"]] + ['train_%s' % m for m in config["metrics"]]


    # 3. Pack all models x repeated CV into a single dictionary
    models_rcvs = dict_cartesian_product(models, rcvs)
    print(len(models_rcvs))


    # Define a wrapper for cross_validate to use with run_parallel
    # This wrapper will cache the results to avoid recomputing them
    @memory.cache
    def cross_validate_wrapper(estimator, cv_test, **kwargs):
        return cross_validate(estimator, X, y, cv=cv_test,
                                scoring=config["metrics"],
                                return_train_score=True)
    
    # 4. Parallel execution with cache
    res = run_parallel(cross_validate_wrapper, models_rcvs, verbose=100, n_jobs=4)

    # 5. Gather results
    res = pd.DataFrame([list(k) + [score[m].mean() for m in metrics_names]
                        for k, score in res.items()],
                    columns=["model", "rep"] + metrics_names)

    res = res.sort_values('test_roc_auc', ascending=False)
        
    # Add stratification SSE to results
    sse = pd.read_csv(config['output_stratification'])
    sse['rep'] = sse['method'].str.cat(sse['seed'].astype(str), sep='-')
    sse = sse[['rep', 'sse', 'sse_weighted']]
    res = pd.merge(res, sse, how="left", on='rep')
    #res.to_csv(config['output_repeatedcv'], index=False)

    # Compute mean by model and rep
    byrep = res.groupby('rep').mean(numeric_only=True).sort_values('test_roc_auc', ascending=False)
    print(byrep.head(20))

    bymod = res.groupby('model').mean(numeric_only=True).sort_values('test_roc_auc', ascending=False)
    print(bymod.head(20))

    # Save results to Excel
    with pd.ExcelWriter(config["output_repeatedcv"]) as writer:
        res.to_excel(writer, sheet_name="All", index=False)
        bymod.to_excel(writer, sheet_name="by_model (repeated CV)")
        byrep.to_excel(writer, sheet_name="by_repetition")
    

################################################################################
# %% Choose a a correctly stratified 5CV split based on min SSE and save it
# ========================================================================

if 'output_cv_test' in config:
    # Chose a 5CV split based on min SSE
   

    # StratifiedKFold with random_state=4 has the better sse in the ten first resample
    # Save it
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=55)
    cv_test = PredefinedSplit([(train, test) for train, test in mskf.split(X, groups_df[factors])])
    #cv_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=4)
    cv_test.to_json(config['output_cv_test'])
    
    cv_stratification_metric(groups_df, factors=factors, cv=cv_test)

    #groups_df['site'] = ['site-%02i' % int(s.split("_")[1]) for s in groups_df['site']]

    ct = pd.crosstab(groups_df['y'], groups_df['site'])
    ct["sum"] = ct.sum(axis=1)
    sum_row = pd.DataFrame([ct.sum(axis=0)], columns=ct.columns)
    ct = pd.concat([ct, sum_row], ignore_index=True)    
    ct.insert(0, 'fold', 'all')
    
    for fold, (train, test) in enumerate(cv_test.split(X, y)):
        ct_fold = pd.crosstab(groups_df.loc[test, 'y'], groups_df.loc[test, 'site'])
        ct_fold["sum"] = ct_fold.sum(axis=1)
        sum_row = pd.DataFrame([ct_fold.sum(axis=0)], columns=ct_fold.columns)
        ct_fold = pd.concat([ct_fold, sum_row], ignore_index=True) 
        ct_fold.insert(0, 'fold', fold)
        ct = pd.concat([ct, ct_fold], axis=0)
    
    print(ct)

################################################################################
# Interpretability: use linear models and group features
# ======================================================
# 
# Focus of some models with a user defined `fit_predict` 
# %% Utils

from nitk.ml_utils.custom_models import Ensure2D

def single_feature_classif(X_train, X_test, y_train, y_test):
    
    make2d = Ensure2D()
    lr = lm.LogisticRegression(fit_intercept=True, class_weight='balanced')
    aucs = np.array([metrics.roc_auc_score(y_test,
                    lr.fit(make2d.transform(X_train[:, j]), y_train).decision_function(make2d.transform(X_test[:, j]))) 
                    for j in range(X_train.shape[1])])
    return aucs

# Mapper: Fit and predict function used in parallel execution
# -----------------------------------------------------------

#@memory.cache
def fit_predict(estimator, X, y, train_idx, test_idx, **kwargs):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    estimator.fit(X_train, y_train)
    
    # 1. Predictions
    y_test_pred_lab = estimator.predict(X_test)
    y_test_pred_proba = estimator.predict_proba(X_test)[:, 1]  # Probability of the positive class
    if hasattr(estimator, 'decision_function'):
        y_test_pred_decision_function = estimator.decision_function(X_test)
    else:
        y_test_pred_decision_function = None # np.nan * np.ones_like(y_test_pred_proba)
    # Decision function for SVMs
    # y_proba = estimator.predict_log_proba(X[test])[:, 1

    # 2. Feature importance
    if isinstance(estimator, Pipeline):
        transformers, head = pipeline_behead(estimator)
        X_train = transformers.transform(X_train)
        X_test = transformers.transform(X_test)
        # prediction_head.fit(Xtr_train, y[train_idx])
    else:
        head = estimator
    
    # 2.1 Coefficients of the prediction head
    coefs_ = get_linear_coefficients(head)
    if coefs_ is not None:
        coefs_ = coefs_.squeeze()
    
    # 2.2 Forward models [Haufe 2014 Forward model] eq. 7:
    if hasattr(head, 'decision_function'):
        #estimator.decision_function(X_test)
        s = head.decision_function(X_test)
        forwd_ = np.dot(X_test.T, s)
    else:
        forwd_ = None
    
    ### 2.3 Individual feature predictive power
    feature_auc = single_feature_classif(X_train, X_test, y_train, y_test)

    # print(y_, y_[test_idx])
    return dict(train_idx=train_idx, test_idx=test_idx,
                y_test_pred_lab=y_test_pred_lab,
                y_test_pred_proba=y_test_pred_proba,
                y_test_pred_decision_function=y_test_pred_decision_function,
                y_test_true_lab=y_test,
                coefs=coefs_, forwd=forwd_, feature_auc=feature_auc,
                estimator=estimator)


# Reducers
# --------
#
# 1. Convert results into DataFrame
# 2. Compute metrics for prediction or feature importance

# 1. Convert result of (parrallel execution) into DataFrame on which we will compute
# metrics

def dict_to_frame(input_dict, keys, base_dict={}):
    """
    Convert a subset of `input_dict` into a one-row pandas DataFrame,
    optionally extended with fixed values from `base_dict`.

    Parameters
    ----------
    input_dict : dict
        A dictionary containing values (typically lists or scalars) to extract.
    keys : list of str
        Keys from `input_dict` to include in the output DataFrame.
    base_dict : dict, optional
        A dictionary of fixed values (scalars or lists) to include in all rows,
        by default {}.

    Returns
    -------
    pd.DataFrame
        A DataFrame constructed from `input_dict[keys]` and `base_dict`. The number of rows
        corresponds to the length of the values in `input_dict[keys[0]]` (assumed consistent).

    Examples
    --------
    >>> input_dict = dict(x1=[1, 2], x2=[10, 20])
    >>> dict_to_frame(input_dict, keys=['x1', 'x2'])
       x1  x2
    0   1  10
    1   2  20

    >>> base_dict = dict(model='regression', fold='1')
    >>> dict_to_frame(input_dict, keys=['x1', 'x2'], base_dict=base_dict)
       model      fold  x1  x2
    0  regression     1   1  10
    1  regression     1   2  20
    """
    output_dict = base_dict.copy()
    output_dict.update({k: input_dict[k] for k in keys})
    return pd.DataFrame(output_dict)


# 2.1 Classification metrics

class ClassificationScorer:
    
    def __init__(self,
                 y_true_lab="y_test_true_lab",
                 y_pred_lab="y_test_pred_lab",
                 y_pred_decision_function="y_test_pred_decision_function",
                 y_pred_proba="y_test_pred_proba",
                 metrics_names=['balanced_accuracy', 'roc_auc']):
    
        self.y_true_lab=y_true_lab
        self.y_pred_lab=y_pred_lab
        self.y_pred_decision_function=y_pred_decision_function
        self.y_pred_proba=y_pred_proba
        self.metrics_names=metrics_names


    def scores(self, y_true_lab, y_pred_lab, y_pred_decision_function, y_pred_proba):
        balanced_accuracy = metrics.balanced_accuracy_score(
            y_true_lab, y_pred_lab)
        # take decision_function when exists else proba
        roc_auc = metrics.roc_auc_score(
            y_true_lab,
            np.where(~np.isnan(y_pred_decision_function),
                        y_pred_decision_function, y_pred_proba))

        return balanced_accuracy, roc_auc

    def predictions_dict_toframe(self, predictions_dict, keys=['test_idx', 'y_test_pred_lab',
                                                        'y_test_pred_decision_function',
                                                        'y_test_pred_proba',
                                                        'y_test_true_lab']):
        
        predictions_df = pd.concat([dict_to_frame(input_dict=val_dict, keys=keys,
                                            base_dict={'model':mod, 'perm':perm, 'fold':fold})
                            for (mod, perm, fold), val_dict in predictions_dict.items()])
        return predictions_df

    def prediction_metrics(self, predictions_df):

        # Compute metrics per model and permutation, and folds
        predictions_metrics_df =  pd.DataFrame(
            [[mod, perm, fold] + list(self.scores(
                predictions_bymodbyperm_df[self.y_true_lab],
                predictions_bymodbyperm_df[self.y_pred_lab],
                predictions_bymodbyperm_df[self.y_pred_decision_function],
                predictions_bymodbyperm_df[self.y_pred_proba]))
            for (mod, perm, fold), predictions_bymodbyperm_df 
            in predictions_df.groupby(['model', 'perm', 'fold'])],
            columns=["model", "perm", 'fold'] + self.metrics_names)

        # Average accross folds
        predictions_metrics_df = \
            predictions_metrics_df.groupby(['model', 'perm']).mean(numeric_only=True).reset_index().sort_values('perm')

        return predictions_metrics_df

    def prediction_metrics_pvalues(self, predictions_metrics_df, permutation_col='perm', true_value='perm-000'):
        
        # Compute p-values and add it to predictions_metrics_df
        predictions_metrics_true_df = predictions_metrics_df[predictions_metrics_df[permutation_col] == true_value]
        #print(predictions_metrics_true_df)
        predictions_metrics_rnd_df = predictions_metrics_df[predictions_metrics_df[permutation_col] != true_value]

        if predictions_metrics_rnd_df.shape[0] > 0:
            #metrics_names = ["balanced_accuracy", "roc_auc"]
            predictions_metrics_pval_df = list()

            for mod, predictions_metrics_rnd_bymod_df in predictions_metrics_rnd_df.groupby('model'):
                balanced_accuracy_h0_mean, roc_auc_h0_mean = predictions_metrics_rnd_bymod_df[self.metrics_names].mean(numeric_only=True)
                balanced_accuracy_h0_std, roc_auc_h0_std = predictions_metrics_rnd_bymod_df[self.metrics_names].std(numeric_only=True)
                balanced_accuracy_h0_pval, roc_auc_h0_pval = (predictions_metrics_rnd_bymod_df[self.metrics_names].values > 
                                                            predictions_metrics_true_df.loc[predictions_metrics_true_df['model']==mod, self.metrics_names].values).sum(axis=0) / nperms

                predictions_metrics_pval_df.append([mod,
                    balanced_accuracy_h0_pval, roc_auc_h0_pval,
                    balanced_accuracy_h0_mean, roc_auc_h0_mean, 
                    balanced_accuracy_h0_std, roc_auc_h0_std])

            predictions_metrics_pval_df = pd.DataFrame(predictions_metrics_pval_df, columns=['model',
                    'balanced_accuracy_h0_pval', 'roc_auc_h0_pval',
                    'balanced_accuracy_h0_mean', 'roc_auc_h0_mean', 
                    'balanced_accuracy_h0_std',  'roc_auc_h0_std'])

            predictions_metrics_true_df = pd.merge(predictions_metrics_true_df, predictions_metrics_pval_df, how='left')

            return predictions_metrics_true_df
        else:
            return None



# 2.2 Feature importance metrics
#
# Feature importance with coefficients of forward model
# Forward-based feature importance [Haufe 2014 Forward model](https://www.sciencedirect.com/science/article/pii/S1053811913010914)
# Forward coefficients: Cov(X, s) where s is the decision function of the prediction head
# s = prediction_head.decision_function(Xtr_train)
# Cov(X, s) = X^T s
# where X is the input data after preprocessing (transformers.fit_transform(Xtr_train, y[train_idx]))

def mean_sd_tval_pval_ci(betas_rep, m0=0):
    """_summary_

    Parameters
    ----------
    betas_rep : array n_repetitions x n_features
        repetition of parameters

    m0 : float
        mean under H0
    Returns
    -------
    _type_
        _description_
    """
    betas_m = np.mean(betas_rep, axis=0)      # sample mean
    betas_s = np.std(betas_rep, ddof=1, axis=0)  # sample standard deviation
    n = betas_rep.shape[0]             # sample size
    df = n - 1
    betas_tval = (betas_m - m0) / betas_s * np.sqrt(n)
    betas_tval_abs = np.abs(betas_tval)
    betas_pval = scipy.stats.t.sf(betas_tval_abs, df) * 2
    #betas_pval_bonf = multipletests(betas_pval, method='bonferroni')[1]
    #betas_pval_fdr = multipletests(betas_pval, method='fdr_bh')[1]
    
    # Critical value for t at alpha / 2:
    t_alpha2 = -scipy.stats.t.ppf(q=0.05/2, df=df, loc=m0)
    ci_low = betas_m - t_alpha2 * betas_s / np.sqrt(n)
    ci_high = betas_m + t_alpha2 * betas_s / np.sqrt(n)
 
    return betas_m, betas_tval, betas_tval_abs,\
        betas_pval, ci_low, ci_high


##
################################################################################
# %% Grouping features by ROIs (utils)
# ====================================
#
# Colinearity is high between features (left, right, CSF, GM) of the same ROI,
# to measure the importance of each ROI. We will group the features by ROI after
# preprocessing and before prediction head and compute the feature importance for each ROI.
# 1. Group input columns by ROI name

def group_by_roi(feature_columns):
    """Group input columns by ROI name, removing optional 'Left' or 'Right' prefixes
    and known suffixes like '_GM_Vol' or '_CSF_Vol'.
    Returns a dictionary where keys are ROI names and values are lists of column names.
    
    Args:
        feature_columns (list): List of column names to group.
    Returns:
        dict: Dictionary with ROI names as keys and lists of column names as values.
    
    Exemple:
        >>> feature_columns_ = [
        ...     "Right Hippocampus_GM_Vol",
        ...     "Left Hippocampus_GM_Vol",
        ...     "Right Hippocampus_CSF_Vol",
        ...     "Left Hippocampus_CSF_Vol",
        ...     "3rd Ventricle_GM_Vol",
        ...     "4th Ventricle_GM_Vol"
        ... ]
        >>> result = group_by_roi(feature_columns_)
        >>> print(result)
        {'Hippocampus': ['Right Hippocampus_GM_Vol', 'Left Hippocampus_GM_Vol', 'Right Hippocampus_CSF_Vol', 'Left Hippocampus_CSF_Vol'], '3rd Ventricle': ['3rd Ventricle_GM_Vol'], '4th Ventricle': ['4th Ventricle_GM_Vol']}
        """
    import re
    from collections import defaultdict
    roi_dict = defaultdict(list)

    for col in feature_columns:
        # Match optional 'Left' or 'Right' prefix
        prefix_match = re.match(r'^(Left|Right)\s+', col)
        base = col

        if prefix_match:
            base = col[len(prefix_match.group(0)):]  # Remove prefix

        # Remove known suffixes if present
        roi_name = re.sub(r'_(GM|CSF)_Vol$', '', base)

        roi_dict[roi_name].append(col)

    return dict(roi_dict)

roi_groups = group_by_roi(feature_columns)
roi_groups = {roi:[feature_columns.index(x) for x in cols] for roi, cols in roi_groups.items()}

pd.DataFrame([[roi, len(subrois), ",".join(subrois)] for roi, subrois in group_by_roi(feature_columns).items()],
            columns=['ROI', 'N_features', 'Features']).to_csv(config['prefix'] + "_roi_groups.csv", index=False)

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

# %% 
# Permutation 0 is without permutations
def permutation(x, random_state=None):
    if random_state == 0:
        return(x)
    if random_state is not None:
        np.random.seed(seed=random_state)
    return np.random.permutation(x)

# %% Prediction and feature importance of Grouping features by ROIs

if 'output_predictions_scores_feature-importance' in config:

    permutation_seed =  pd.read_csv(config['input_permutations']).perm.values
    permutation_seed = [0]
    nperms = len(permutation_seed) - 1
    # df = pd.read_excel("/home/ed203246/git/2025_spetiton_rlink_predict_response_anat/03_classif_rois/models/20_ml-classifLiResp_predictions_scores_feature-importance_v-20250708.xlsx", sheet_name='predictions')
    # perms = pd.DataFrame(dict(perm=[int(perm.split('-')[1]) for perm in df.perm.unique()]))
    # perms.to_csv(config['input_permutations'])

    # perumtation (permute y) x folds


    # Load the CV test split
    cv_test = PredefinedSplit(json_file=config['input_cv_test'])

    # {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
    cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i'  % fold):
        (X, permutation(y, perm), train_index, test_index)
        for perm in permutation_seed
        for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
    print(cv_test_dict_Xy.keys())


    from sklearn.cluster import FeatureAgglomeration

    # Models
    mlp_param_grid = {"hidden_layer_sizes":
                        [
                        (100, ), (50, ), (25, ), (10, ), (5, ),       # 1 hidden layer
                        (100, 50, ), (50, 25, ), (25, 10,), (10, 5, ), # 2 hidden layers
                        (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, ), # 3 hidden layers
                        ],
                        "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}

    from sklearn.neural_network import MLPClassifier

    models = {
        'model-lrl2_resid-age+sex+site':
        make_pipeline(residualizer_estimator, preprocessing.StandardScaler(),
                GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                            {'C': 10. ** np.arange(-3, 1)},
                            cv=cv_val, n_jobs=5, scoring='accuracy')), # 'balanced_accuracy'
        # 'model-grpRoiLr+lrl2_resid-age+sex+site':
        # make_pipeline(residualizer_estimator, preprocessing.StandardScaler(),
        #         GroupFeatureTransformer(roi_groups,  LogisticRegressionTransformer(fit_intercept=False, class_weight='balanced', C=0.01)),
        #         preprocessing.StandardScaler(),
        #         GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
        #                     {'C': 10. ** np.arange(-3, 1)},
        #                     cv=cv_val, n_jobs=5)),
        # 'model-grpRoiPca+lrl2_resid-age+sex+site':
        # make_pipeline(residualizer_estimator, preprocessing.StandardScaler(),
        #         GroupFeatureTransformer(roi_groups,  "pca"),
        #         preprocessing.StandardScaler(),
        #         GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
        #                     {'C': 10. ** np.arange(-3, 1)},
        #                     cv=cv_val, n_jobs=5)),
        # 'model-grpRoiMean+lrl2_resid-age+sex+site':
        # make_pipeline(residualizer_estimator, preprocessing.StandardScaler(),
        #         GroupFeatureTransformer(roi_groups,  "mean"),
        #         preprocessing.StandardScaler(),
        #         GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
        #                     {'C': 10. ** np.arange(-3, 1)},
        #                     cv=cv_val, n_jobs=5)),
        'model-grpRoiLda+lrl2_resid-age+sex+site':
        make_pipeline(residualizer_estimator, #preprocessing.StandardScaler(),
                GroupFeatureTransformer(roi_groups,  "lda"),
                preprocessing.StandardScaler(),
                GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                            {'C': 10. ** np.arange(-3, 1)},
                            cv=cv_val, n_jobs=5, scoring='balanced_accuracy')),
        'model-grpRoiLdaClust2+lrl2_resid-age+sex+site':
        make_pipeline(residualizer_estimator, #preprocessing.StandardScaler(),
                GroupFeatureTransformer(roi_groups,  "lda"),
                FeatureAgglomeration(n_clusters=2),
                preprocessing.StandardScaler(),
                GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                            {'C': 10. ** np.arange(-3, 1)},
                            cv=cv_val, n_jobs=5)),
        'model-grpRoiLdaClust3+lrl2_resid-age+sex+site':
        make_pipeline(residualizer_estimator, #preprocessing.StandardScaler(),
                GroupFeatureTransformer(roi_groups,  "lda"),
                FeatureAgglomeration(n_clusters=3),
                preprocessing.StandardScaler(),
                GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                            {'C': 10. ** np.arange(-3, 1)},
                            cv=cv_val, n_jobs=5)),
        'model-mlp_resid-age+sex+site':
        make_pipeline(residualizer_estimator, 
                preprocessing.MinMaxScaler(),
                GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=200, tol=0.01),
                            param_grid=mlp_param_grid,
                            cv=cv_val, n_jobs=5))
        }

    features_names = {
        'model-lrl2_resid-age+sex+site':feature_columns,
        'model-mlp_resid-age+sex+site':feature_columns,
        'model-grpRoiLda+lrl2_resid-age+sex+site':roi_groups.keys(),
        'model-grpRoiLdaClust2+lrl2_resid-age+sex+site': ["clust-%02i" % i for i in range(2)],
        'model-grpRoiLdaClust3+lrl2_resid-age+sex+site': ["clust-%02i" % i for i in range(3)]
    }

    models_cv = dict_cartesian_product(models, cv_test_dict_Xy)

    # Fit models

    res_cv = run_parallel(fit_predict, models_cv, verbose=50)
    #res_cv = run_sequential(fit_predict, models_cv, verbose=50)
    # res_cv[[k for k in res_cv.keys()][0]].keys()

    # Classifications metrics
    reducer = ClassificationScorer()
    predictions_df = reducer.predictions_dict_toframe(res_cv)
    predictions_metrics_df = reducer.prediction_metrics(predictions_df)
    print(predictions_metrics_df)

    predictions_metrics_pvalues_df = reducer.prediction_metrics_pvalues(predictions_metrics_df)
    print(predictions_metrics_pvalues_df)

    # Feature importance

    # labels features depending en model see: features_names[mod]
    features_df = pd.concat([dict_to_frame(input_dict=val_dict,
        keys=['coefs', 'forwd', 'feature_auc'],
        base_dict={'model':mod, 'perm':perm, 'fold':fold, 'feature':features_names[mod]})
        for (mod, perm, fold), val_dict in res_cv.items()])

    #features_stat_names = ['coefs_tval', 'forwd_tval']
    #nfolds_cv_test = 5

    def features_statistics(features_df, feature_name_col='feature', feature_importance_cols={'coefs':0, 'forwd':0, 'feature_auc':0.5}):

        # For all mod, perm concatenate values accross folds store if in a features[(mod, perm), feature_importance_col]
        from collections import defaultdict
        features = defaultdict(list)

        # For each feature importance statistic append across folds
        # dict[(mod, perm), feature_importance_col] = [vals0, vals1, ...]
        for (mod, perm, fold), df in features_df.groupby(['model', 'perm', 'fold']):
            df = df.set_index(feature_name_col)
            for feature_importance_col in feature_importance_cols.keys():
                features[(mod, perm), feature_importance_col].append(df[feature_importance_col]) 

        # Feature importance folds-wise statistics
        features_stats = list()
        for ((mod, perm), feature_importance_col), vals in features.items():
            #print(mod, perm, feature_importance_col, vals.shape)
            vals = pd.concat(vals, axis=1)
            for i in range(vals.shape[0]):
                features_stats.append([mod, perm, vals.index[i], feature_importance_col] +\
                    list(mean_sd_tval_pval_ci(vals.iloc[i, :].values, m0=feature_importance_cols[feature_importance_col])))

        features_stats = pd.DataFrame(features_stats, columns=['model', 'perm', feature_name_col, 'stat', 'mean', 'tval', 'tval_abs', 'pval', 'ci_low', 'ci_high'])

        return features_stats

    features_stats = features_statistics(features_df)

    stat_pval = {}
    # Split by models and compute corrected p-values
    for (mod, stat), df in features_stats.groupby(['model', 'stat']):
        #if mod == 'model-grpRoiLda+lrl2_resid-age+sex+site' and stat=='feature_auc':
        #if mod == 'model-grpRoiLda+lrl2_resid-age+sex+site' and stat=='forwd':
        #    break
        print(mod, stat)
        true_df = df[df['perm']=='perm-000'].copy().set_index('feature')
        rnd_df = df[df['perm']!='perm-000']
        
        # Corrected p-values
        true_df['pval_bonferroni'] = multipletests(true_df['pval'], method='bonferroni')[1]
        true_df['pval_fdr_bh'] = multipletests(true_df['pval'], method='fdr_bh')[1]

        # Statistics under H0
        mean_rnd = pd.concat([df_.set_index('feature')['mean'] for perm, df_ in rnd_df.groupby('perm')], axis=1)    
        true_df['mean_h0'] = mean_rnd.mean(axis=1)
        tval_rnd = pd.concat([df_.set_index('feature')['tval'] for perm, df_ in rnd_df.groupby('perm')], axis=1)
        true_df['tval_h0'] = tval_rnd.mean(axis=1)
        # if mod == 'model-grpRoiLda+lrl2_resid-age+sex+site' and stat=='forwd':
        #     break
        # #tval_true_ = true_df.loc[tval_rnd.index == "Hippocampus", 'tval']
        # tval_rnd_ = tval_rnd.loc[tval_rnd.index == "Hippocampus", :].values.ravel()
        # perms_ = np.array([int(perm.split("-")[1]) for perm, _ in rnd_df.groupby('perm')])
        # n_ok = np.where(tval_rnd_ >= 2.80743991428207)[0]
        # ok = np.where(tval_rnd_ < 2.80743991428207)[0]
        # perms_ = perms_[np.sort(list(np.random.choice(ok, 955, replace=False)) + list(np.random.choice(n_ok, 45, replace=False)))]
        # perms_df = pd.DataFrame(dict(perm=perms_))
        # perms_df.to_csv("permutations.csv", index=False)
        
        
        # Compute randomize p-values
        true_df['mean_pval_rnd'] = mean_rnd.gt(true_df['mean'], axis=0).sum(axis=1) / mean_rnd.shape[1]
        true_df['tval_pval_rnd'] = tval_rnd.gt(true_df['tval'], axis=0).sum(axis=1) / tval_rnd.shape[1]
        
        # Westfall & Young FWER cor p-values

        stat_max_rnd = tval_rnd.max(axis=0).values #stat_max_rnd.shape: (nperms, )
        stat_values = true_df['tval'].values # stat_values.shape: (nfeatures, )
        # Reshape for broadcasting: (nfeatures, 1) vs (1, nperms)
        pval_tmax_fwer = (stat_max_rnd > stat_values[:, None]).sum(axis=1) / nperms # pval_tmax_fwer.shape (nfeatures, )
        true_df['tval_pval_tmax_fwer'] = pd.Series(pval_tmax_fwer, index=true_df.index)

        true_df = true_df.reset_index()
        true_df.sort_values('pval', inplace=True)
        stat_pval[(mod, stat)] = true_df

    with pd.ExcelWriter(config['output_predictions_scores_feature-importance']) as writer:#, mode="a", if_sheet_exists="replace") as writer:
        predictions_metrics_pvalues_df.to_excel(writer, sheet_name='predictions_metrics_pvalues', index=False)
        predictions_df.to_excel(writer, sheet_name='predictions', index=False)
        for (mod, stat), df in stat_pval.items():
            sheet_name = '__'.join([mod, stat])
            # print(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


################################################################################
# %% Interpretation of Group ROIs: LDA coef and forward correlation
# =================================================================
# Get group LDA coefficients and final forward correlation with residualized data
# Necessary to interpret the effect between signal in features and decision function of the prediction head

if 'output_RoiGrdLda_coefs' in config:
    print("Interpretation of Group ROIs: LDA coef and forward correlation")

    rois_selected = ['Amygdala', 'Hippocampus', 'Inf Lat Vent', 'MPoG postcentral gyrus medial segment']
    feats_selected = [feature_columns[feat_idx] for roi in rois_selected for feat_idx in roi_groups[roi]]
    #[[roi, feature_columns[feat_idx]] for roi in rois_selected for feat_idx in roi_groups[roi]]

    # Load the CV test split
    cv_test = PredefinedSplit(json_file=config['input_cv_test'])

    permutation_seed = [0]
    # {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
    cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i'  % fold):
        (X, permutation(y, perm), train_index, test_index)
        for perm in permutation_seed
        for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
    print(cv_test_dict_Xy.keys())

    models =  {'model-grpRoiLda+lrl2_resid-age+sex+site':
        make_pipeline(residualizer_estimator, #preprocessing.StandardScaler(),
                GroupFeatureTransformer(roi_groups,  "lda"),
                preprocessing.StandardScaler(),
                GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                            {'C': 10. ** np.arange(-3, 1)},
                            cv=cv_val, n_jobs=5, scoring='balanced_accuracy'))}
        
    models_cv = dict_cartesian_product(models, cv_test_dict_Xy)


    def correlation(X, Y):
        scaler = preprocessing.StandardScaler()
        Xsc = scaler.fit_transform(X)
        Ysc = scaler.fit_transform(Y)
        #print(Xsc.T @ Ysc)
        return Xsc.T @ Ysc / Xsc.shape[0]  # Covariance matrix

    transformers_coefs = list()
    forward_corr = pd.DataFrame()

    for (mod, perm, fold), (estimator, X, y, train_idx, test_idx) in models_cv.items():
        print(mod, perm, fold)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        estimator.fit(X_train, y_train)
        X_test_res = estimator.steps[0][1].transform(X_test)

        # Correlation between features and decision function
        s = estimator.decision_function(X_test)
        corr_ = correlation(X_test_res, s.reshape(-1, 1))
        len(feature_columns)
        forward_corr_ = pd.DataFrame(dict(model=mod, perm=perm, fold=fold, feature=feature_columns, Xy_corr=corr_.ravel()))
        forward_corr = pd.concat([forward_corr, forward_corr_], axis=0)
        
        transformers = estimator.steps[-3][1].column_transformer_.transformers_
        for roi, transformer, columns in transformers:
            #if roi in ['Amygdala', 'Hippocampus', 'Inf Lat Vent', 'MPoG postcentral gyrus medial segment']:
            for i, col in enumerate(columns):
                transformers_coefs.append([mod, perm, fold, roi, feature_columns[col], transformer.coef_.ravel()[i]])

    #len(transformers_coefs)

    forward_corr_stat = forward_corr.groupby(['model', 'feature']).describe().reset_index().sort_values(by=['feature'])

    transformers_coefs = pd.DataFrame(transformers_coefs, columns=['model', 'perm', 'fold', 'roi', 'feature', 'LDAcoef'])
    transformers_coefs_stat = transformers_coefs.groupby(['model', 'roi', 'feature']).describe().reset_index().sort_values(by=['roi', 'feature'])

    summary_stat = pd.merge(transformers_coefs_stat, forward_corr_stat)
    summary_stat = summary_stat.loc[summary_stat[('roi', '')].isin(rois_selected),]

    with pd.ExcelWriter(config['output_RoiGrdLda_coefs']) as writer:#, mode="a", if_sheet_exists="replace") as writer:
        summary_stat.to_excel(writer, sheet_name='SUMMARY_stat')#, index=False)
        transformers_coefs_stat.to_excel(writer, sheet_name='RoiGrdLda_stat')#, index=False)
        transformers_coefs.to_excel(writer, sheet_name='RoiGrdLda_coefs')#, index=False)
        forward_corr_stat.to_excel(writer, sheet_name='forward_corr_stat')#, index=False)
        forward_corr.to_excel(writer, sheet_name='forward_corr')#, index=False)


    # Plot correlation matrix of residualized data
    estimator = clone(models['model-grpRoiLda+lrl2_resid-age+sex+site'])
    X_train, X_test = X, X
    y_train, y_test = y, y
    estimator.fit(X_train, y_train)
    X_train_res = estimator.steps[0][1].transform(X_train)
    X_train_res_df = pd.DataFrame(X_train_res, columns=feature_columns)

    R = X_train_res_df[feats_selected].corr()
    sns.heatmap(R, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 6}, fmt=".2f",)

    plt.savefig(config['output_feature_correlation_matrix']+".svg", format="svg")
    plt.close()
    plt.show()


# if False:
#     ################################################################################
#     # Classification Without CSF
#     # =============================

#     feature_columns_nocsf = [col for col in feature_columns if not 'CSF' in col]


#     # Check that X == [Z + X[feature_columns]]
#     X_nocsf = get_X(data, feature_columns_nocsf, print_log=print).values
#     #X_[:, (csf_indices)] *= -1
#     X_nocsf = residualizer_estimator.pack(Z, X_nocsf)
#     assert X_nocsf.shape[1] == Z.shape[1] + len(feature_columns_nocsf)

#     # Load the CV test split
#     cv_test = PredefinedSplit(json_file=config['input_cv_test'])

#     permutation_seed = [0]
#     # {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
#     cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i'  % fold):
#         (X_nocsf, permutation(y, perm), train_index, test_index)
#         for perm in permutation_seed
#         for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
#     print(cv_test_dict_Xy.keys())

#     roi_groups_nocsf = group_by_roi(feature_columns_nocsf)
#     roi_groups_nocsf = {roi:[feature_columns_nocsf.index(x) for x in cols] for roi, cols in roi_groups_nocsf.items()}


#     models = {
#             'model-lrl2_resid-age+sex+site':
#             make_pipeline(residualizer_estimator, preprocessing.StandardScaler(),
#                     GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
#                                 {'C': 10. ** np.arange(-3, 1)},
#                                 cv=cv_val, n_jobs=5, scoring='accuracy')), # 'balanced_accuracy'
#             'model-grpRoiLda+lrl2_resid-age+sex+site':
#             make_pipeline(residualizer_estimator, #preprocessing.StandardScaler(),
#                     GroupFeatureTransformer(roi_groups_nocsf,  "lda"),
#                     preprocessing.StandardScaler(),
#                     GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
#                                 {'C': 10. ** np.arange(-3, 1)},
#                                 cv=cv_val, n_jobs=5, scoring='balanced_accuracy')),
#             'model-mlp_resid-age+sex+site':
#             make_pipeline(residualizer_estimator, 
#                     preprocessing.MinMaxScaler(),
#                     GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=200, tol=0.01),
#                                 param_grid=mlp_param_grid,
#                                 cv=cv_val, n_jobs=5))
#             }

#     features_names = {
#         'model-lrl2_resid-age+sex+site':feature_columns,
#         'model-mlp_resid-age+sex+site':feature_columns,
#         'model-grpRoiLda+lrl2_resid-age+sex+site':roi_groups.keys()
#     }

#     models_cv = dict_cartesian_product(models, cv_test_dict_Xy)
#     res_cv = run_parallel(fit_predict, models_cv, verbose=50)

#     # Classifications metrics
#     reducer = ClassificationScorer()
#     predictions_df = reducer.predictions_dict_toframe(res_cv)
#     predictions_metrics_df = reducer.prediction_metrics(predictions_df)
#     print(predictions_metrics_df)

#     """
#                                         model      perm  balanced_accuracy   roc_auc
#     0  model-grpRoiLda+lrl2_resid-age+sex+site  perm-000           0.624365  0.667725
#     1            model-lrl2_resid-age+sex+site  perm-000           0.602579  0.637275
#     2             model-mlp_resid-age+sex+site  perm-000           0.519603  0.637857
#     """


################################################################################
# %% Permutation based feature importance
# =======================================
# The permutation importance is defined to be the difference between the
# baseline metric and metric from permutating the feature column.
# important features have large positive value

rois_selected = ['Amygdala', 'Hippocampus', 'Inf Lat Vent', 'MPoG postcentral gyrus medial segment']
feats_selected = [feature_columns[feat_idx] for roi in rois_selected for feat_idx in roi_groups[roi]]
#[[roi, feature_columns[feat_idx]] for roi in rois_selected for feat_idx in roi_groups[roi]]

# Load the CV test split
cv_test = PredefinedSplit(json_file=config['input_cv_test'])

permutation_seed = [0]
# {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i'  % fold):
    (X, permutation(y, perm), train_index, test_index)
    for perm in permutation_seed
    for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
print(cv_test_dict_Xy.keys())

models =  {'model-grpRoiLda+lrl2_resid-age+sex+site':
    make_pipeline(residualizer_estimator, #preprocessing.StandardScaler(),
            GroupFeatureTransformer(roi_groups,  "lda"),
            preprocessing.StandardScaler(),
            GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                        {'C': 10. ** np.arange(-3, 1)},
                        cv=cv_val, n_jobs=5, scoring='balanced_accuracy'))}
    
models_cv = dict_cartesian_product(models, cv_test_dict_Xy)


def correlation(X, Y):
    scaler = preprocessing.StandardScaler()
    Xsc = scaler.fit_transform(X)
    Ysc = scaler.fit_transform(Y)
    #print(Xsc.T @ Ysc)
    return Xsc.T @ Ysc / Xsc.shape[0]  # Covariance matrix

transformers_coefs = list()
forward_corr = pd.DataFrame()

from sklearn.inspection import permutation_importance

feature_importances = list()
roi_importances = list()

for (mod, perm, fold), (estimator, X, y, train_idx, test_idx) in models_cv.items():
    print(mod, perm, fold)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    estimator.fit(X_train, y_train)
    
    # Features importance
    residualizer, predictor = pipeline_split(estimator, 1)
    X_test_res = residualizer.transform(X_test)
    feat_imp = permutation_importance(predictor, X_test_res, y_test,
                           scoring='roc_auc',
                           n_repeats=5,
                           n_jobs=5,
                           random_state=0)
    feature_importances.append(feat_imp['importances_mean'])

    # Roi importance
    residualizer, predictor = pipeline_behead(estimator)
    X_test_res = residualizer.transform(X_test)
    feat_imp = permutation_importance(predictor, X_test_res, y_test,
                           scoring='roc_auc',
                           n_repeats=5,
                           n_jobs=5,
                           random_state=0)
    roi_importances.append(feat_imp['importances_mean'])


feature_importances = np.array(feature_importances)
assert feature_importances.shape[1] == len(feature_columns)
fi_mean = feature_importances.mean(axis=0)
fi_std = feature_importances.std(axis=0)
fi_z = fi_mean / fi_std

feature_importances_df = pd.DataFrame(dict(feature=feature_columns,
     importances_z=fi_z,                  
     importances_mean=fi_mean,
     importances_std=fi_std))

feature_importances_df.sort_values('importances_z', ascending=False, inplace=True)

roi_importances = np.array(roi_importances)
assert roi_importances.shape[1] == len(roi_groups.keys())
ri_mean = roi_importances.mean(axis=0)
ri_std = roi_importances.std(axis=0)
ri_z = ri_mean / ri_std

roi_importances_df = pd.DataFrame(dict(roi=roi_groups.keys(),
     importances_z=ri_z,                  
     importances_mean=ri_mean,
     importances_std=ri_std))
roi_importances_df.sort_values('importances_z', ascending=False, inplace=True)


feature_importances_df.to_csv('/tmp/output_feature_importances.csv', index=False)

X_permuted = np.arange(25).reshape(5, 5)
col_idx = np.array([0, 2])
shuffling_idx = np.array([4, 3, 2, 1, 0])
X_permuted[shuffling_idx[:, np.newaxis], col_idx]

X_permuted[:, col_idx] = X_permuted[shuffling_idx[:, np.newaxis], col_idx]

#feature_importances_order = np.argsort(fi_z)[::-1]
#fi_z[feature_importances_order]
#[feature_columns[i] for i in feature_importances_order]


    feat_imp['importances_mean'].shape
    feat_imp['importances_std'].shape
    feat_imp['importances'].shape
    feat_imp['importances'].mean(axis=1).shape
    feat_imp['importances'].mean(axis=1) == feat_imp['importances_mean']
    #.keys()
    #['importances_mean', 'importances_std', 'importances'])

# %%
