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
config['prefix'] = "031_classif_rois_resampling-scheme"
config['output_repeatedcv'] = os.path.join(config['output_models'], config['prefix'] + "_repeatedcv.xlsx")

# Set output paths
config['log_filename'] = config['prefix'] + ".log"
config['cachedir'] = config['prefix'] + ".cachedir"

# Print log function
if 'log_filename' not in config:
    print_log = create_print_log(config['log_filename'])
    print_log('###################################################################')
    print_log('## %s' % datetime.now().strftime("%Y-%m-%d %H:%M"))
    print_log(config)


# Create cachedir
from joblib import Memory
if 'cachedir' not in config:
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
# %% 3. CV Stratification scheme
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
# %% 4. Repeated CV using cross_validate
# ======================================
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
    models = make_models(n_jobs_grid_search=1, cv_val=cv_val,
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
    res = run_parallel(cross_validate_wrapper, models_rcvs, verbose=100, n_jobs=5)

    # 5. Gather results
    res = pd.DataFrame([list(k) + [score[m].mean() for m in metrics_names]
                        for k, score in res.items()],
                    columns=["model", "rep"] + metrics_names)

    res = res.sort_values('test_roc_auc', ascending=False)
        
    # Add stratification SSE to results
    # sse = pd.read_csv(config['output_stratification'])
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
# %% 5. Choose a correctly stratified 5CV split based on min SSE and save it
# ==========================================================================

if 'cv_test' in config:
    # Chose a 5CV split based on min SSE
   

    # StratifiedKFold with random_state=4 has the better sse in the ten first resample
    # Save it
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=55)
    cv_test = PredefinedSplit([(train, test) for train, test in mskf.split(X, groups_df[factors])])
    #cv_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=4)
    cv_test.to_json(config['cv_test'])
    # cv_test = PredefinedSplit(json_file=config['cv_test'])

    cv_stratification_metric(groups_df, factors=factors, cv=cv_test)

    #groups_df['site'] = ['site-%02i' % int(s.split("_")[1]) for s in groups_df['site']]

    ct = pd.crosstab(groups_df['y'], groups_df['site'])
    ct["sum"] = ct.sum(axis=1)
    sum_row = pd.DataFrame([ct.sum(axis=0)], columns=ct.columns)
    ct = pd.concat([ct, sum_row], ignore_index=True)
    ct.insert(0, 'fold', 'all')
    
    cv_df = data[['participant_id', 'site', 'y', 'age', 'sex']].copy()
    cv_df["fold"] = None
    
    for fold, (train, test) in enumerate(cv_test.split(X, y)):
        ct_fold = pd.crosstab(groups_df.loc[test, 'y'], groups_df.loc[test, 'site'])
        ct_fold["sum"] = ct_fold.sum(axis=1)
        sum_row = pd.DataFrame([ct_fold.sum(axis=0)], columns=ct_fold.columns)
        ct_fold = pd.concat([ct_fold, sum_row], ignore_index=True) 
        ct_fold.insert(0, 'fold', fold)
        ct = pd.concat([ct, ct_fold], axis=0)
        cv_df.loc[test, "fold"] = fold 
    print(ct)

    cv_df.to_csv(config['cv_test'].replace('.json', '.csv'), index=False)


