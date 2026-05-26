
################################################################################
# %% 1. Initialization 
# ====================
import os.path
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from config import config, cv_val
from ml_utils import PredefinedSplit
from mulm.residualizer.residualizer import Residualizer, ResidualizerEstimator


################################################################################
# %% CONFIGURATION
# ================

# Load the CV test split
#
#OUTPUT = "reports/classification_reports.xlsx"


################################################################################
# %% Load Data
# ============

data = pd.read_csv(config['input_data'])

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
# %% Stratifed K-Fold CV
# ======================

groups_df = pd.DataFrame(dict(y=y, site=data['site']))
factors = ['y', 'site']
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=55)
splits_index_mskf = [(train_index, test_index) for train_index, test_index in
                    mskf.split(X, groups_df[factors])]


# Save the CV test split
cv_test = PredefinedSplit([(train, test) for train, test in splits_index_mskf])
cv_test.to_json('stratified-5cv_REMOVE-MANUALLY.json')
# Manually rename stratified-5cv_REMOVE-MANUALLY.json' => 'stratified-5cv.json' to avoid accidentally overwriting the file with a different split.

# Save the CV test split with participant IDs instead of indices
cv_test_participant_id = PredefinedSplit([(data.iloc[train]["participant_id"], data.iloc[test]["participant_id"])
                                          for train, test in splits_index_mskf])
cv_test_participant_id.to_json('stratified-5cv_participant_id_REMOVE-MANUALLY.json')



################################################################################
# %% Check the CV splits with stored JSON files
# =============================================

cv_test_stored         = PredefinedSplit(json_file='stratified-5cv.json')
cv_test_new            = PredefinedSplit(json_file='stratified-5cv_REMOVE-MANUALLY.json')
cv_test_participant_id = PredefinedSplit(json_file='stratified-5cv_participant_id.json')

# ── Consistency checks ────────────────────────────────────────────────────────

# Map participant_id → row index for the third check
pid_to_idx = {pid: idx for idx, pid in enumerate(data['participant_id'])}

splits_stored = list(cv_test_stored.split(X, y))
splits_new    = list(cv_test_new.split(X, y))
splits_pid    = list(cv_test_participant_id.split(X, y))

print("\n── cv_test_stored vs cv_test_new ────────────────────────────────────")
for fold, ((tr1, te1), (tr2, te2)) in enumerate(zip(splits_stored, splits_new)):
    tr_eq = np.array_equal(sorted(tr1), sorted(tr2))
    te_eq = np.array_equal(sorted(te1), sorted(te2))
    print(f"  Fold {fold}: train={'OK' if tr_eq else 'DIFF'}  test={'OK' if te_eq else 'DIFF'}")

print("\n── cv_test_stored vs cv_test_participant_id ─────────────────────────")
for fold, ((tr_idx, te_idx), (tr_pid, te_pid)) in enumerate(zip(splits_stored, splits_pid)):
    tr_from_pid = sorted(pid_to_idx[p] for p in tr_pid)
    te_from_pid = sorted(pid_to_idx[p] for p in te_pid)
    tr_eq = np.array_equal(sorted(tr_idx), tr_from_pid)
    te_eq = np.array_equal(sorted(te_idx), te_from_pid)
    print(f"  Fold {fold}: train={'OK' if tr_eq else 'DIFF'}  test={'OK' if te_eq else 'DIFF'}")

