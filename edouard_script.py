################################################################################
# Imports
# -------

# %%

import pandas as pd, numpy as np
import sys, os, os.path, tempfile, time, logging, json
# import matplotlib.pyplot as plt, seaborn as sns # plotting

# univariate statistics
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms

# Dataset
from sklearn.datasets import make_classification

# from itertools import product

# Joblib
from joblib import Parallel, delayed
# from joblib import Memory
from joblib import cpu_count

# Metrics
import sklearn.metrics as metrics
# %%
# Resampling
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
# from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

# Set pandas display options
# pd.set_option('display.max_colwidth', None)  # No maximum width for columns
# pd.set_option('display.width', 1000)  # Set the total width to a large number

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"

################################################################################
# Read config file (Parameters)

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

# Import models
sys.path.append(os.path.dirname(config["models_path"]))
root, _ = os.path.splitext(os.path.basename(config["models_path"]))

config['log_filename'] = os.path.splitext(config_file)[0] + ".log"
config['output_filename'] = os.path.splitext(config_file)[0] + "_scores_corrected_no_balanced_weights.csv"
config['stratification_sse'] = os.path.splitext(config_file)[0] + "_stratification-sse.csv"
config['cv5test'] = os.path.splitext(config_file)[0] + "_cv5test_mskf.json"

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
print(np.shape(y))
print(y[:10])
print(type(y))
################################################################################
# X: Input Data
# Select Input = dataframe - (target + drop + residualization)
# %%

input_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]

X = get_X(data, input_columns, print_log=print_log)
X = X.values
# %%
################################################################################
# Z: Residualization data
# %%
residualization_formula = False

if config['residualization']:  
    X, residualizer_estimator, residualization_formula = \
        get_residualizer(data, X, residualization_columns=config['residualization'],
                    print_log=print_log)

print(residualization_formula) #age+sex+site
# %%
################################################################################
# Repeated CV Validation scheme. Compute SSE of sub-group proportion 
# between original data and test sample to ensure good stratification for
# factors
# %%
from nitk.ml_utils.cross_validation import PredefinedSplit

df = pd.DataFrame(dict(y=y, site=data['site']))
factors = ['y', 'site']
print(df)

n_splits_test = 5
cv_test = StratifiedKFold(n_splits=n_splits_test,
                          shuffle=True, random_state=42)

# %%
n_splits_val = 5
cv_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=42)
# %%
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

"""
we calculate the nb of rows (subjects) that have each pair of (y, site) possible,
dividing it by the total number of subjects/rows
we have the proportion for each unique tuple (y, site) of subjects
then, we compute how far off it is from the proportion of each tuple (y, site) in the
whole dataset 
We want a stratification that ensures that the 5 CV folds have the closest 
distribution to the entire dataset as possible
"""

# github https://github.com/trent-b/iterative-stratification
# paper https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

all_idx = np.arange(len(df))
cv_id = PredefinedSplit([[all_idx, all_idx] for i in range(5)])
assert cv_stratification_metric(df, factors=factors, cv=cv_id) == 0 # the SSE btw the whole dataset and the whole dataset should be equal to zero

# testing 100 random seeds with 2 stratification methods
sse = []
rcvs = dict()

for seed in range(100):
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits_index_mskf = [(train_index, test_index) for train_index, test_index in
                         mskf.split(X, df[factors])]
    # warning with MultilabelStratifiedKFold: here it works because site and y are catagorical but it would require binarization
    # if another (continuous) variable was added to the list of factors (like age)
    mskf = PredefinedSplit(splits_index_mskf)
    rcvs["mskf-%i" % seed] = mskf

    sse_ = cv_stratification_metric(df, factors=factors, cv=mskf)
    sse.append(["mskf", seed, sse_])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    #splits_index_skf = [test_index for train_index, test_index in skf.split(X, y)]
    rcvs["skf-%i" % seed] = skf
    sse_ = cv_stratification_metric(df, factors=factors, cv=skf)
    sse.append(["skf", seed, sse_])

sse = pd.DataFrame(sse, columns=['method', 'seed', 'sse'])


if not os.path.isfile(config['cv5test']):
    # Chose a 5CV split based on min SSE
    print(sse.groupby('method').mean())
    """
    mskf and skf performed similarly:
            seed       sse
    method                
    mskf    49.5  0.104003
    skf     49.5  0.104159

    => choose skf
    """

    sse.to_csv(config['stratification_sse'], index=False)

    sse[sse.method=="skf"].iloc[:10, :]
    first_ten_seeds=sse[sse.method=="skf"].iloc[:10, :]
    seed_with_min_sse = first_ten_seeds.loc[first_ten_seeds["sse"].idxmin(), "seed"]
    # first_ten_seeds_mskf=sse[sse.method=="mskf"].iloc[:10, :]

    """
       method  seed       sse
    1     skf     0  0.081951
    3     skf     1  0.123264
    5     skf     2  0.115742
    7     skf     3  0.113581
    9     skf     4  0.080669
    11    skf     5  0.090485
    13    skf     6  0.104369
    15    skf     7  0.112598
    17    skf     8  0.078463
    19    skf     9  0.082032

    method  seed       sse
    0    mskf     0  0.076341
    2    mskf     1  0.125219
    4    mskf     2  0.115845
    6    mskf     3  0.088354
    8    mskf     4  0.121734
    10   mskf     5  0.083182
    12   mskf     6  0.115488
    14   mskf     7  0.139668
    16   mskf     8  0.114015
    18   mskf     9  0.115422
    """

    # StratifiedKFold with random_state=4 has the better sse in the ten first resample
    # Save it
    cv_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=4) # change random_state to seed_with_min_sse
    cv_stratification_metric(df, factors=factors, cv=cv_test)

    PredefinedSplit([(train, test) for train, test in cv_test.split(X, y)]).to_json(config['cv5test'])
    cv_test = PredefinedSplit(json_file=config['cv5test'])
    cv_stratification_metric(df, factors=factors, cv=cv_test)

    ct = pd.crosstab(df['y'], df['site'])
    ct.insert(0, 'fold', 'all')
    for fold, (train, test) in enumerate(cv_test.split(X, y)):
        ct_fold = pd.crosstab(df.loc[test, 'y'], df.loc[test, 'site'])
        ct_fold.insert(0, 'fold', fold)
        ct = pd.concat([ct, ct_fold], axis=0)
    print(ct) # display nb of elements (participants) for each label (0 or 1) for each site

# %%   
cv_test = PredefinedSplit(json_file=config['cv5test'])
print(cv_stratification_metric(df, factors=factors, cv=cv_test))

################################################################################
# Execution function
# %%
def run_sequential(func, iterable_dict, memory=None,  verbose=0,
                   *args, **kwargs):

    start_time = time.time()
    res = {k:func(*v, verbose=verbose) for k, v in iterable_dict.items()}

    if verbose > 0:
        print('Sequential execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))
    
    return res


def run_parallel(func, iterable_dict, memory=None,  verbose=0,
                   *args, **kwargs):

    parallel = Parallel(n_jobs=cpu_count(only_physical_cores=True))

    start_time = time.time()
    res = parallel(delayed(func)(*v, verbose=verbose)
                   for k, v in iterable_dict.items())

    if verbose > 0:
        print('Parallel execution, Elapsed time: \t%.3f sec' %
              (time.time() - start_time))

    return {k:r for k, r in zip(iterable_dict.keys(), res)}

################################################################################
# Repeated CV using cross_validate
# --------------------------------
#
# See `cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#>`_`
# Choose `scoring functions <https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers>`_ 

# %% Models

# Do not parallelize grid search
models = make_models(n_jobs_grid_search=1, cv_val=cv_val,
                     residualization_formula=residualization_formula,
                     residualizer_estimator=residualizer_estimator)

print("Models (N=%i)" % len(models), models.keys())

metrics_names = ['test_%s' % m for m in config["metrics"]]


# %% Pack all models x repeated CV into a single dictionary
from nitk.python_utils import dict_cartesian_product
from sklearn.model_selection import cross_validate

models_rcvs = dict_cartesian_product(models, rcvs) #rcvs is a dict with keys 'mskf-'+str(seed) and values are tr/te arrays for 5-fold CV
print(len(models_rcvs))
print(models_rcvs)
# %% 

# config['cachedir'] is /neurospin/signatures/
# 2025_spetiton_rlink_predict_response_anat/models/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4/supervised_classification_config/cachedir
# memory = Memory(config['cachedir'], verbose=0)

# @memory.cache
# %% 
def cross_validate_wrapper(estimator, cv_test, **kwargs):
    return cross_validate(estimator, X, y, cv=cv_test,
                            scoring=config["metrics"],
                            return_train_score=True, n_jobs=1)

# %%
print(config['output_filename'])
print(metrics_names)
# %% Parallel execution with cache
if not os.path.isfile(config['output_filename']):
    res = run_parallel(cross_validate_wrapper, models_rcvs, verbose=1)
    res = pd.DataFrame([list(k) + [score[m].mean() for m in metrics_names]
                        for k, score in res.items()],
                    columns=["model", "rep"] + metrics_names)
    res = res.sort_values('test_roc_auc', ascending=False)
    res.to_csv(config['output_filename'], index=False)
    print(res)

else : 
    print("tests already run... reading output file...")
    res = pd.read_csv(config["output_filename"])

# config['output_filename']:
# /neurospin/signatures/2025_spetiton_rlink_predict_response_anat/models/
# study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4/supervised_classification_config_scores.csv

sse['rep'] = sse['method'].str.cat(sse['seed'].astype(str), sep='-')
print(res)
res = pd.merge(res, sse, how="left")
print(res)
print(config['output_filename'])
res.to_csv(config['output_filename'], index=False)
quit()

print(res.groupby('rep')['test_roc_auc'].mean().sort_values(ascending=False))
print(res[res["rep"]=="skf-8"]['test_roc_auc'].mean()) # 0.615 no balanced_weights for classification, 0.58 with balanced_weights
print(res[res["rep"]=="skf-4"]['test_roc_auc'].mean()) # 0.654 no balanced_weights for classification, 0.63 with balanced_weights

quit()

"""
rep
skf-4      0.659702
mskf-4     0.656336
mskf-47    0.651792
skf-47     0.644046
skf-76     0.641540
"""


# %%
