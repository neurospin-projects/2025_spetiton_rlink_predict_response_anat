################################################################################
# %% Imports
# ----------

# System
import sys
import os
import os.path
import time
import json
from datetime import datetime


# Scientific python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests

# Univariate statistics
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms

# from itertools import product

# Models
from sklearn.base import clone
# from sklearn.decomposition import PCA
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import StackingClassifier

# Metrics
import sklearn.metrics as metrics

# Resampling
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
#from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import sklearn.linear_model as lm
from sklearn.compose import ColumnTransformer

# Set pandas display options
pd.set_option('display.max_colwidth', None)  # No maximum width for columns
pd.set_option('display.width', 1000)  # Set the total width to a large number

################################################################################
# %% Config dictionary
# --------------------

config = {
    # Working directory
    "WD": "/home/ed203246/git/2025_spetiton_rlink_predict_response_anat/03_classif_rois/",
    # Input data
    "input_data" : "./data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4.csv",
    # Paths
    "LD_LIBRARY_PATH": ["/home/ed203246/git/nitk"],
    #"models_path": "./classification_models.py"
    # Output directories
    'output_models': "./models",
    'output_reports': "./reports",
    # Variables
    "target": "y",
    "target_remap": {"NR":0, "PaR":0, "GR":1},
    "residualization": ["age", "sex", "site"],
    "drop": ["participant_id"],
    "metrics": ["accuracy", "balanced_accuracy", "roc_auc"],
}

# Initialize WD
os.chdir(config["WD"])

    
# LD_LIBRARY_PATH
if "LD_LIBRARY_PATH" in config:
    for path in config["LD_LIBRARY_PATH"]:
        sys.path.append(path) # Add to system path


################################################################################
# %% Additional imports (from nitk)
# ---------------------------------

#import nitk
#from nitk.sys_utils import import_module_from_path, create_print_log
#from nitk.pandas_utils.dataframe_utils import expand_key_value_column
#from nitk.pandas_utils.dataframe_utils import describe_categorical
#from nitk.ml_utils.dataloader_table import get_y, get_X
#from nitk.ml_utils.residualization import get_residualizer
#from nitk.jobs_utils import run_sequential, run_parallel
#from nitk.ml_utils.cross_validation import PredefinedSplit
#from nitk.python_utils import dict_cartesian_product
#from nitk.ml_utils.sklearn_utils import pipeline_behead, pipeline_split, get_linear_coefficients
#from nitk.ml_utils.custom_models import GroupFeatureTransformer
#from nitk.pandas_utils.dataframe_utils import describe_categorical
#from nitk.sys_utils import create_print_log

#from nitk.ml_utils.iterstrat.ml_stratifiers import MultilabelStratifiedKFold

n_splits_val = 5
cv_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=42)


###################################################################################
# %% Classification models
# ------------------------

# import numpy as np

# # Models
# from sklearn.base import clone
# from sklearn.decomposition import PCA
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import StackingClassifier

# from sklearn.pipeline import make_pipeline


# from sklearn.model_selection import GridSearchCV
# # from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold
# from sklearn import preprocessing
# from sklearn.pipeline import make_pipeline



# ################################################################################
# # %% Utils functions
# # ------------------

# def get_y(data, target_column, remap_dict=None, print_log=print):
#     """_summary_

#     Parameters
#     ----------
#     data : _type_
#         _description_
#     target_column : _type_
#         _description_
#     remap_dict: dict
#         remap target, defualts None
#     print_log : _type_, optional
#         _description_, by default print

#     Returns
#     -------
#     _type_
#         _description_

#     Yields
#     ------
#     _type_
#         _description_
#     """
#     print_log('\n# y (target)\n"%s", counts:' % target_column)
#     print_log(describe_categorical(data[target_column]))

#     if remap_dict:
#         y = data[target_column].map(remap_dict)
#         print_log('After remapping, counts:')
#         print_log(describe_categorical(y))
#     else:
#         y = data[target_column]
#     return y


# def get_X(data, input_columns, print_log=print):
#     """Get input Data. Perform dummy codings for categorical variables

#     Parameters
#     ----------
#     data : DataFrame
#         Input dataFrame
#     input_columns : list
#         input columns
#     print_log: callable function, default print

#     Returns
#     -------
#         pd.DataFrame: input Data
#     """
#     X = data[input_columns]
#     ncol = X.shape[1]

#     print_log('\n# X (Input data)')
#     print_log(X.describe(include='all').T)

#     categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
#     if categorical_cols:
#         print_log("\nCategorical columns:", categorical_cols)
#         for v in categorical_cols:
#             print_log(v, describe_categorical(data[v]))
#         X = pd.get_dummies(X, dtype=int)
#         print_log('\nAfter coding')
#         print_log(X.describe(include='all').T)
#         print_log('%i dummies variable created' %  (X.shape[1] - ncol))

#     return X

