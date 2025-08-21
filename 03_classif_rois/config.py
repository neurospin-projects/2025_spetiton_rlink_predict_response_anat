################################################################################
# %% Imports
# ----------

# System
import sys
import os
# import os.path
# import time
# import json
# from datetime import datetime


# # Scientific python
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats
# from statsmodels.stats.multitest import multipletests

# Univariate statistics
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms

# from itertools import product

# Models
# from sklearn.base import clone
# from sklearn.decomposition import PCA
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import StackingClassifier

# # Metrics
# import sklearn.metrics as metrics

# # Resampling
# # from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_validate
# # from sklearn.model_selection import cross_val_predict
# # from sklearn.model_selection import train_test_split
# # from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
# #from sklearn import preprocessing

# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import make_pipeline
# from sklearn import preprocessing
# import sklearn.linear_model as lm
# from sklearn.compose import ColumnTransformer

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
    "cv_test" : "stratified-5cv.json"
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




