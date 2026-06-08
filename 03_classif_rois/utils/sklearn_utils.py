"""
Sklearn Utilities
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score,
                             matthews_corrcoef)

from utils.plot_utils import plot_roc

# ------------------------------------------------------------------------------
#  Pipelines utils
# ------------------------------------------------------------------------------

def pipeline_split(pipeline, step=-1):
    """Split pipeline into two pipelines body[:step] and head[step:].
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object.
    step : int
        The step where to split the pipeline
    Returns
    -------
    body : Pipeline
        A new Pipeline object containing "step" fist steps.
    head : Pipeline
        A new Pipeline object containing the remaining steps.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Estimator must be a Pipeline instance")


    body = Pipeline(pipeline.steps[:step])
    head = Pipeline(pipeline.steps[step:])
    
    return body, head
    
    
def pipeline_behead(pipeline):
    """Separate preprocessing transformers from the prediction head of a pipeline estimator.
    This function assumes that the last step of the pipeline is the prediction head
    (e.g., a classifier or regressor) and all previous steps are preprocessing steps.
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object.
    Returns
    -------
    transformers : Pipeline
        A new Pipeline object containing only the preprocessing steps.
    prediction_head : object
        The last step of the original pipeline, which is the prediction head (e.g., classifier
        Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression())])
    >>> transformers, prediction_head = pipeline_behead(pipe)
    >>> # Apply preprocessing to data
    >>> X_preprocessed = transformers.transform(X)
    >>> # Use prediction head on preprocessed data
    >>> prediction_head.predict(X_preprocessed)
    array([0, 0, 1, 1])
    >>> pipe.fit(X, y)
    >>> pipe.predict(X)
    array([0, 0, 1, 1])
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Estimator must be a Pipeline instance")

    #pipeline = clone(pipeline)  # Clone the estimator to avoid modifying the original
    preprocessing_steps = pipeline.steps[:-1]
    # Get the last step (the predictor)
    _, prediction_head = pipeline.steps[-1]
    # Create a new pipeline with only the preprocessing steps
    transformers = Pipeline(preprocessing_steps)
    
    return transformers, prediction_head

def get_predictor(estimator):
    """Unwrap a fitted meta-estimator and return the underlying predictor.

    Handles any object exposing ``best_estimator_`` (e.g. GridSearchCV,
    RandomizedSearchCV). Returns the estimator itself when it is already
    a plain fitted predictor.
    """
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_
    return estimator


def get_coef(estimator):
    """Extract a 1-D coefficient array from a fitted predictor.

    Tries ``coef_`` (linear models) then ``feature_importances_`` (tree
    models). Raises ``AttributeError`` if neither attribute exists.
    """
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        return coef[0] if coef.ndim > 1 else coef
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    raise AttributeError(
        f"{type(estimator).__name__} exposes neither 'coef_' nor "
        "'feature_importances_' — cannot extract coefficients."
    )

class Ensure2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # stateless transformer

    def transform(self, X):
        # Case 1: pandas Series (1D) → convert to DataFrame
        if isinstance(X, pd.Series):
            return X.to_frame()

        # Case 2: pandas DataFrame (2D) → return as-is
        if isinstance(X, pd.DataFrame):
            return X

        # Case 3: NumPy array
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

def oof_arrays_from_cv(vals_cv: list, cv, X: np.ndarray, y: np.ndarray, split: str= 'test') -> np.ndarray:
    """Reconstruct a full out-of-fold array from per-fold results.

    Inverse of the accumulation loop used in cross-validation: places each
    fold's values back at the test indices to produce a single array aligned
    with the original sample order.

    Parameters
    ----------
    vals_cv : list of ndarray, one per fold, each of shape (n_test_i, ...)
    cv      : CV splitter (same instance used during fitting)
    X       : feature matrix (used only for cv.split)
    y       : target array  (used only for cv.split)
    split   : str, either 'test' or 'train' indicating which indices to reconstruct
    Returns
    -------
    out : ndarray of shape (n_samples, ...) with fold results placed at their
          original test indices
    """
    if split is 'test':
        n_samples = np.sum([len(te) for _, te in cv.split(X, y)])
    else:
        n_samples = np.sum([len(tr) for tr, _ in cv.split(X, y)])
    first    = vals_cv[0]
    out      = np.zeros((n_samples, *first.shape[1:]), dtype=first.dtype)
    
    for (tr, te), vals in zip(cv.split(X, y), vals_cv):
        if split is 'test':
            out[te] = vals
        else:
            out[tr] = vals
    return out


def classification_report_cv(X: np.ndarray,
                             y: np.ndarray,
                             estimators: list,
                             cv,
                             as_one_row: bool = False) -> pd.DataFrame:
    """
    Evaluate a set of pre-fitted CV estimators out-of-fold and return a tidy
    metrics DataFrame.

    Each metric is reported with two strategies:

    - **Average-of-folds** (avg_folds ± std_folds): metric computed within each
      fold then averaged across folds — identical to what ``cross_val_score``
      reports.
    - **Pooled OOF** (pooled_oof): out-of-fold predictions concatenated across
      all folds, then scored once on the full vector — identical to what
      ``cross_val_predict`` followed by a metric call produces.

    Recall is broken down **per class label** (``Recall (class 0)``,
    ``Recall (class 1)``, …) so that sensitivity and specificity are reported
    separately instead of being collapsed into a single binary recall.

    Parameters
    ----------
    X          : ndarray (n_samples, n_features) — raw feature matrix
    y          : ndarray (n_samples,) — true class labels
    estimators : list of fitted pipelines, one per CV fold
    cv         : CV splitter used during fitting (must match the one used to
                 produce ``estimators``)
    as_one_row : bool, default False
        If True, flatten ``metrics_df`` into a single row with MultiIndex
        columns ``(metric, stat)`` — convenient for stacking results across
        models into one summary table.

    Returns
    -------
    metrics_df : pd.DataFrame
        If ``as_one_row=False`` (default): one row per metric, columns:
        ``metric``, ``avg_folds``, ``std_folds``, ``se_folds``,
        ``pooled``, ``pval_pooled``, ``pval_fold``, ``fold_values``.
        - ``se_folds`` = std / √n_folds.
        - ``pval_pooled``: one-sided p-value on pooled OOF — binomial test
          (H0: score ≤ 0.5) for balanced accuracy and per-class recalls;
          Mann-Whitney U (H0: AUC ≤ 0.5) for ROC-AUC; NaN otherwise.
        - ``pval_fold``: one-sided one-sample t-test against 0.5 on the
          per-fold values — available for balanced accuracy, ROC-AUC, and
          per-class recalls; NaN for Precision, F1, MCC.
        If ``as_one_row=True``: single-row DataFrame with MultiIndex columns
        ``(metric, stat)``; ``fold_values`` column is excluded.
    """
    
    # 1. Average-of-folds / "score-then-average": compute the metric within each fold,
    # then average across folds.

    print("\n━━━  Average-of-folds (AOF) Classification Report ━━━")

    y_pred_cv  = np.empty(len(y), dtype=int)
    y_proba_cv = np.empty(len(y))
    ba_fold, auc_fold, prec_fold, f1_fold, mcc_fold = [], [], [], [], []
    rec_cls_fold = []   # per-fold recall for each class: list of (n_classes,) arrays

    for (_, te), pipe in zip(cv.split(X, y), estimators):
        y_pred_cv[te]  = pipe.predict(X[te])
        y_proba_cv[te] = pipe.predict_proba(X[te])[:, 1]
        ba_fold.append(balanced_accuracy_score(y[te], y_pred_cv[te]))
        auc_fold.append(roc_auc_score(y[te], y_proba_cv[te]))
        prec_fold.append(precision_score(y[te], y_pred_cv[te], zero_division=0))
        rec_cls_fold.append(recall_score(y[te], y_pred_cv[te], average=None, zero_division=0))
        f1_fold.append(f1_score(y[te], y_pred_cv[te], zero_division=0))
        mcc_fold.append(matthews_corrcoef(y[te], y_pred_cv[te]))

    classes    = np.unique(y)
    n_folds    = len(ba_fold)
    N_test     = len(y)
    
    # P-values on average-of-folds
    t, auc_pval_fold = ttest_1samp(auc_fold, popmean=0.5, alternative='greater')
    t, ba_pval_fold  = ttest_1samp(ba_fold,  popmean=0.5, alternative='greater')
    rec_pval_fold = [ttest_1samp([r[i] for r in rec_cls_fold], popmean=0.5, alternative='greater')[1]
                     for i in range(len(classes))]
    
    # 2. Pooled OOF / "average-then-score": concatenate all OOF predictions, then compute the metric once on the full vector.

    rec_pooled = recall_score(y, y_pred_cv, average=None, zero_division=0)  # (n_classes,)
    ba_pooled  = balanced_accuracy_score(y, y_pred_cv)
    auc_pooled = roc_auc_score(y, y_proba_cv)

    # p-values on pooled OOF
    ba_pval_pooled  = scipy_stats.binomtest(
        k=int(ba_pooled * N_test), n=N_test, p=0.5, alternative='greater').pvalue
    labels = np.unique(y)
    if len(labels) == 2:
        auc_pval_pooled = scipy_stats.mannwhitneyu(
            y_proba_cv[y == labels[0]], y_proba_cv[y == labels[1]], alternative='greater').pvalue
        # auc_pval_pooled = min(auc_pval_pooled, 1 - auc_pval_pooled)
    else:
        auc_pval_pooled = np.nan  # Mann-Whitney U test is not defined for multiclass AUC
    rec_pval_pooled = [
        scipy_stats.binomtest(
            k=int(rec_pooled[i] * np.sum(y == cls)),
            n=int(np.sum(y == cls)), p=0.5, alternative='greater').pvalue
        for i, cls in enumerate(classes)
    ]

    rows = [
        ("ROC-AUC",           auc_fold,  auc_pooled,                                     auc_pval_pooled, auc_pval_fold),
        ("Balanced Accuracy", ba_fold,   ba_pooled,                                      ba_pval_pooled,  ba_pval_fold),
        ("Precision",         prec_fold, precision_score(y, y_pred_cv, zero_division=0), np.nan,          np.nan),
        *[(f"Recall (class {cls})", [r[i] for r in rec_cls_fold], rec_pooled[i], rec_pval_pooled[i], rec_pval_fold[i])
          for i, cls in enumerate(classes)],
        ("F1",                f1_fold,   f1_score(y, y_pred_cv, zero_division=0),        np.nan,          np.nan),
        #("MCC",               mcc_fold,  matthews_corrcoef(y, y_pred_cv),                np.nan,          np.nan),
    ]
    metrics_df = pd.DataFrame(
        [(name, np.mean(folds), np.std(folds), np.std(folds) / np.sqrt(n_folds), pooled, pval_pooled, pval_fold, folds)
         for name, folds, pooled, pval_pooled, pval_fold in rows],
        columns=["metric", "avg_folds", "std_folds", "se_folds", "pooled", "pval_pooled", "pval_fold", "fold_values"],
    )

    if as_one_row:
        metrics_df = (metrics_df
                      .set_index("metric")
                      .drop(columns=["fold_values"], errors='ignore')
                      .stack().to_frame().T
                      .reset_index(drop=True))

    return metrics_df

# %%
def drop_indices_from_folds(folds, drop_indices):
    """
    Remove indices in drop_idx from cross-validation folds and
    shift remaining indices to account for the removed rows.

    Parameters
    ----------
    folds    : list of [train_indices, test_indices] pairs (arrays or lists)
    drop_idx : list/array of integer indices to remove

    Returns
    -------
    new_folds : same structure as folds, with drop_idx removed and
                remaining indices shifted down accordingly.
    """
    drop_indices = sorted(drop_indices)

    def shift(idx):
        new = []
        for i in idx:
            if i in drop_indices:
                continue  # remove
            shift_down = sum(d < i for d in drop_indices)
            new.append(i - shift_down)
        return np.array(new)

    return [[shift(tr), shift(te)] for tr, te in folds]

# def drop_indices_from_folds(folds, drop_indices):
#     """Remove a set of sample indices from every fold and re-index to the new dataset.

#     After removal the returned indices refer to positions in the compacted dataset
#     (original dataset minus the dropped rows). For every kept index ``i`` its new
#     position is ``i - #{j in drop_indices : j < i}``, computed via
#     ``np.searchsorted``.

#     Parameters
#     ----------
#     folds : list of (array-like, array-like)
#         Cross-validation folds as returned by ``PredefinedSplit.predefined_splits``
#         or ``PredefinedSplit.split()``. Each element is a ``(train_idx, test_idx)``
#         pair of integer arrays (positions in the *original* dataset).
#     drop_indices : array-like of int
#         Sample indices to remove (e.g. outlier row positions in X / y).

#     Returns
#     -------
#     list of (np.ndarray, np.ndarray)
#         Cleaned folds whose indices are valid positions in the new dataset
#         ``X[keep_mask]`` / ``y[keep_mask]``.

#     Examples
#     --------
#     >>> keep_mask  = ~is_outlier
#     >>> X_clean, y_clean = X[keep_mask], y[keep_mask]
#     >>> outlier_idx = np.where(is_outlier)[0]
#     >>> clean_folds = drop_indices_from_folds(cv_test.predefined_splits, outlier_idx)
#     >>> cv_clean = PredefinedSplit(predefined_splits=clean_folds)
#     """
#     import numpy as np
#     import warnings

#     drop_sorted = np.sort(np.asarray(drop_indices, dtype=int))
#     drop_set    = set(drop_sorted.tolist())

#     def _reindex(idx_array):
#         """Keep non-dropped indices and shift each one down by the number of
#         dropped indices that precede it."""
#         kept = np.array([i for i in idx_array if i not in drop_set], dtype=int)
#         # searchsorted with side='left' gives #{j in drop_sorted : j < i}
#         shift = np.searchsorted(drop_sorted, kept, side="left")
#         return kept - shift

#     clean_folds = []
#     for fold_idx, (tr, te) in enumerate(folds):
#         tr_clean = _reindex(tr)
#         te_clean = _reindex(te)
#         if len(tr_clean) == 0 or len(te_clean) == 0:
#             warnings.warn(
#                 f"Fold {fold_idx} has {'no training' if len(tr_clean) == 0 else 'no test'} "
#                 f"samples after dropping {len(drop_set)} indices.", UserWarning
#             )
#         clean_folds.append((tr_clean, te_clean))
#     return clean_folds


# %%
if __name__ == "__main__":
    folds = [
        [np.array([0, 1, 2, 3]), np.array([4, 5, 6])],
        [np.array([0, 4, 5, 6]), np.array([1, 2, 3])],
    ]
    new_folds = drop_indices_in_cv(folds, drop_idx=[5])  # your case
# index 5 removed, all indices > 5 shifted down by 1