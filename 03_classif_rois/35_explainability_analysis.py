"""
Explainability Analysis Pipeline  —  Lithium Response (Binary Classification)
=============================================================================
Inputs:
    Xdf : pd.DataFrame  — clinical feature matrix
    y   : array-like    — binary lithium response (0/1)

Model:
    StandardScaler -> GridSearchCV(LogisticRegression(fit_intercept=False,
                                                      class_weight='balanced'),
                                   {'C': 10. ** np.arange(-3, 1)},
                                   cv=cv_val, n_jobs=5, scoring='accuracy')

Outer evaluation:
    cv_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

Public API:
    metrics                              = plot_classification_report(X, y, cv_results['estimator'], cv_test)
    shap_cv, X_trn_cv, shap_stats_df = shap_analysis(cv_results['estimator'], X, y, features, cv_test)
    perm_imps_cv, perm_stats_df          = permutation_importance_analysis(cv_results['estimator'], X, y, features, cv_test)
    coefs_boot, boot_stats_df            = bootstrap_stability(model, X, y, features)
    fig, ax                              = comparison_dashboard({"SHAP": shap_stats_df, ...})


# Methods Test shorten

To identify brain regions driving lithium-response prediction, we evaluated feature importance using three complementary methods, all computed out-of-fold across the five test folds of the outer cross-validation so that explanations reflect generalisation rather than training-set fit. Computing importance scores out-of-fold also yields fold-level estimates whose variability can be characterised: for each method and feature, we tested whether the score was significantly greater than zero using a one-sample t-test across folds, and report importance as −log₁₀(p), a scale-free statistic that is directly comparable across methods.
The first two methods follow a permute-and-predict logic, quantifying the expected marginal contribution of each feature. (i) SHAP values (Lundberg & Lee, 2017) assign to each feature its Shapley-value contribution to the log-odds of each individual prediction, providing a locally faithful additive decomposition; per fold, we averaged absolute SHAP values across test samples. (ii) Permutation importance (Breiman, 2001; Pedregosa et al., 2011; 30 repeats per fold) quantifies the drop in balanced accuracy when a single feature is randomly shuffled on the held-out test set; per fold, we averaged the drop across repeats. For both methods, fold-level scores were submitted to the one-sample t-test described above to yield −log₁₀(p) importance. Because marginal-contribution methods are sensitive to inter-feature correlation — correlated features share explanatory credit, attenuating individual scores — we complement them with a third approach. (iii) Haufe forward patterns (Haufe et al., 2014) transform the discriminative weights of the linear classifier into a forward, generative model — the covariance between each input feature and the model's decision function — thereby mitigating multicollinearity-induced attenuation of importance estimates, and were likewise summarised as −log₁₀(p) across folds.
Finally, features were ranked by their median −log₁₀(p) across the three methods, and the top 20% are reported.



Methods Text Long:


## Feature importance: Clinical predictors of outcome

To identify clinical variables driving lithium-response prediction, we evaluated
feature importance within the multivariate logistic-regression model using four
complementary methods.
We organised them into two methodological families that probe distinct aspects of the model.

**Perturbation-based methods (no refitting).** Three methods explain the 
already-fitted model by perturbing the inputs and measuring the effect on its 
predictions, without retraining. They are commonly grouped under the umbrella of post-hoc,
perturbation-based explainability (Molnar, 2022; Covert et al., 2021) and 
were all computed out-of-fold across the five test folds of the outer cross-validation, 
so that explanations reflect generalisation rather than training-set fit.

(i) *SHAP values* (Lundberg & Lee, 2017), assign to each feature its Shapley-value 
contribution to the log-odds of each individual prediction, providing a locally faithful, 
additive decomposition. We report the mean absolute SHAP value across out-of-fold samples as a global importance score.

(ii) *Permutation importance* (Breiman, 2001), implemented with scikit-learn (Pedregosa et al., 2011) 
(30 repeats per fold), quantifies the drop in balanced accuracy when a single feature is randomly shuffled 
on the held-out test set, thus reflecting the model's reliance on each feature.

(iii) *Haufe forward patterns* (Haufe et al., 2014) transform the discriminative
weights of the linear classifier into a forward, generative model — the forward
pattern defined as the covariance between each input feature and the model's
decision function. This correction mitigates the multicollinearity-induced
attenuation of importance estimates that arises because correlated features are
interchangeable in the model and share credit. It also distinguishes genuinely
predictive features from suppressor variables, features that improve model
performance by capturing useful but outcome-non-specific signal (e.g., age)
without being directly predictive of the outcome themselves.


**Resampling-based stability (with refitting).** A complementary family of methods 
does not perturb inputs but rather perturbs the *training sample* itself, 
refitting the model on each resample to assess the stability of its parameters 
under sampling variability. This is the resampling-based stability framework 
(Meinshausen & Bühlmann, 2010; Sauerbrei & Schumacher, 1992), often referred 
to as *bootstrap stability* when the perturbation is generated by non-parametric bootstrap.
We drew B = 200 bootstrap resamples (with replacement, preserving sample size), 
refitted the full pipeline — including standardisation and grid-search-tuned regularisation — on each resample,
and recorded the standardised logistic-regression coefficients.
For every feature we report the bootstrap mean, standard deviation, 
95 % percentile confidence interval, and sign-consistency rate 
(the percentage of bootstrap replicates in which the coefficient retained the 
sign of its bootstrap mean). Features whose effect is both large in magnitude 
and consistent in sign across resamples can be considered robust predictors, 
whereas features whose sign flips between resamples are flagged as unstable 
regardless of their average magnitude.

The two families answer different questions: perturbation-based methods 
ask *how much does the fitted model rely on this feature?*, whereas resampling-based 
stability asks *how reproducible is this feature's contribution under sampling variability?* 
We therefore report all four scores side-by-side, normalised to a common scale, 
in a comparison dashboard. Convergence across methods is taken as evidence of a 
genuine clinical predictor; divergence is flagged and discussed.


Results WIP

Out of 71 brain regions, the top 20% (n = 15) ranked by median -log10(p) across the three methods were: 4th Ventricle, Cerebellum White Matter, MPoG postcentral gyrus medial segment, Ventral DC, OpIFG opercular part of the inferior frontal gyrus, PHG parahippocampal gyrus, OFuG occipital fusiform gyrus, Amygdala, LOrG lateral orbital gyrus, TTG transverse temporal gyrus, Thalamus Proper, TMP temporal pole, AOrG anterior orbital gyrus, Cuneus, and Accumbens Area. Nine of these 15 regions reached significance (p < 0.05) consistently across all three methods (4th Ventricle, Cerebellum White Matter, MPoG, Ventral DC, OpIFG, OFuG, Amygdala, TMP temporal pole, Cuneus), and are considered the most robust predictors. Three additional regions (LOrG, AOrG, Accumbens Area) were significant for SHAP and Permutation importance but not for the Haufe forward pattern (p > 0.10), suggesting their contribution to the decision boundary may partly reflect multicollinearity rather than a direct generative link to the outcome. The PHG parahippocampal gyrus was significant for SHAP and Haufe but not for permutation importance (p = 0.175), possibly indicating that its contribution is detectable at the individual level but modest in terms of average accuracy drop. Inter-method agreement was low to moderate: Spearman correlations of -log10(p) across all features were r = 0.49 (SHAP vs Permutation, p < 0.001), r = 0.26 (SHAP vs Haufe, p = 0.027), and r = 0.22 (Permutation vs Haufe, p = 0.068), consistent with the distinct sensitivities of perturbation-based and forward-model approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn import linear_model as lm, preprocessing
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer

import shap
import scipy.stats
import warnings
warnings.filterwarnings("ignore")

from utils.sklearn_utils import (pipeline_behead, get_predictor, get_coef,
                                 oof_arrays_from_cv)

from utils.sklearn_utils import classification_report_cv
from statsmodels.stats.multitest import multipletests


def mean_sd_se_tval_pval_ci(x, m0=0, alpha=0.05, m=None, sd=None, se=None,
                            alternative='two-sided'):
    """Compute per-feature summary statistics and a one-sample t-test against `m0`.

    Treats each column of `x` as an independent feature and each row as a
    repetition (e.g., a cross-validation fold, bootstrap resample, or
    permutation). Assumes observations within a column are i.i.d. and
    approximately normal; for moderately large `n` the t-test is robust to
    mild departures from normality.

    Parameters
    ----------
    x : array-like of shape (n_repetitions, n_features)
        Observations. Rows are repetitions, columns are features.
    m0 : float, default=0
        Mean under the null hypothesis H0: E[x] = m0.
    alpha : float, default=0.05
        Significance level used for the two-sided confidence interval
        (coverage = 1 - alpha).
    m : array-like or None, default=None
        Pre-computed mean per feature. Estimated from `x` if None.
    sd : array-like or None, default=None
        Pre-computed standard deviation per feature. Estimated from `x` if None.
    se : array-like or None, default=None
        Pre-computed standard error per feature. Derived from `sd` if None.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Defines the alternative hypothesis.
        'two-sided': H1: E[x] != m0
        'less':      H1: E[x] < m0
        'greater':   H1: E[x] > m0

    Returns
    -------
    DataFrame of shape (7, n_features) with index
    [mean, sd, se, tval, pval, ci_low, ci_high].
    Columns are ``x.columns`` when `x` is a DataFrame, otherwise 0-based integers.
    """
    columns = x.columns if isinstance(x, pd.DataFrame) else None
    x = np.asarray(x)
    n = x.shape[0]
    df = n - 1

    if m  is None: m  = np.mean(x, axis=0)
    if sd is None: sd = np.std(x, ddof=1, axis=0)
    if se is None: se = sd / np.sqrt(n)

    tval = (m - m0) / se
    if alternative == 'two-sided':
        pval = 2 * scipy.stats.t.sf(np.abs(tval), df)
    elif alternative == 'greater':
        pval = scipy.stats.t.sf(tval, df)
    elif alternative == 'less':
        pval = scipy.stats.t.cdf(tval, df)
    else:
        raise ValueError(f"alternative must be 'two-sided', 'less', or 'greater', got {alternative!r}")

    t_crit  = scipy.stats.t.isf(alpha / 2, df)
    ci_low  = m - t_crit * se
    ci_high = m + t_crit * se

    _, pval_corr, _, _ = multipletests(pval, method="fdr_bh")

    return pd.DataFrame(
        [m, sd, se, tval, pval, pval_corr, ci_low, ci_high],
        index=["mean", "sd", "se", "tval", "pval", "pval_corr", "ci_low", "ci_high"],
        columns=columns,
    )

# %%
# ==============================================================================
# Feature importance functions
# ==============================================================================

# ------------------------------------------------------------------------------
# HAUF Forward pattern
# ------------------------------------------------------------------------------


def haufe_feature_importance(estimators: list,
                             X: np.ndarray,
                             y: np.ndarray,
                             features: list,
                             cv,
                             normalise: bool = False) -> tuple:
    """
    Compute Haufe activation patterns (forward model) out-of-fold.

    Corrects for the fact that linear classifier weights are not directly
    interpretable as feature importances when features are correlated.
    The activation pattern a = Cov(X, s) where s is the decision-function
    output — Haufe et al. (2014) NeuroImage, eq. 7.

    Parameters
    ----------
    estimators : list of fitted pipelines (one per CV fold)
    X          : raw (unscaled) feature matrix
    y          : binary response array (used only for cv.split)
    features   : ordered list of feature names
    cv         : CV splitter used during fitting

    Returns
    -------
    haufe_cv        : list of ndarray (n_features,), one per fold — raw values
    haufe_cv_df : DataFrame (n_folds, n_features) — |Haufe| per fold;
                      columns = feature names
    haufe_stat_cv_df : DataFrame (7, n_features) from mean_sd_se_tval_pval_ci
    """
    print("\n━━━  Haufe Forward Model  (out-of-fold)  ━━━")
    haufe_cv = []

    for fold, ((_, te), estimator) in enumerate(zip(cv.split(X, y), estimators), 1):
        if isinstance(estimator, Pipeline):
            body, head = pipeline_behead(estimator)
            X_trn_te   = body.transform(X[te])
        else:
            X_trn_te = X[te]
            head     = estimator
        predictor = get_predictor(head)
        # Haufe 2014 eq. 7: a = Cov(X, s) ≈ X_centered^T @ s / N
        # StandardScaler ensures X_trn_te is already centred.
        s = predictor.decision_function(X_trn_te)
        haufe_cv.append(X_trn_te.T @ s / len(s))
        print(f"   fold {fold} done")

    # haufe_cv_df = pd.DataFrame(np.abs(np.vstack(haufe_cv)), columns=features)
    haufe_cv_df = pd.DataFrame(np.vstack(haufe_cv), columns=features)

    if normalise:
        haufe_cv_df = haufe_cv_df / haufe_cv_df.mean(axis=0).sum()

    haufe_stat_cv_df = mean_sd_se_tval_pval_ci(haufe_cv_df)
    haufe_stat_cv_df.loc["rank"] = haufe_stat_cv_df.loc["pval"].rank(ascending=True)

    print("\n  Haufe activation pattern summary:")
    print(haufe_stat_cv_df.round(4))
    return haufe_cv, haufe_cv_df, haufe_stat_cv_df


# ------------------------------------------------------------------------------
# SHAP — LinearExplainer
# ------------------------------------------------------------------------------

def shap_analysis(estimators: list,
                  X: np.ndarray,
                  y: np.ndarray,
                  features: list,
                  cv,
                  normalise: bool = False) -> tuple:
    """
    Compute out-of-fold SHAP values with LinearExplainer across the CV loop.
    Each fold's test samples are scaled with that fold's scaler and explained
    with that fold's best LogisticRegression — no leakage.

    Parameters
    ----------
    estimators : list of fitted pipelines (one per CV fold)
    X          : raw (unscaled) feature matrix
    y          : binary response array (used only for cv.split)
    features   : ordered list of feature names
    cv         : CV splitter used during fitting

    Returns
    -------
    shap_cv    : list of ndarray (n_test_i, n_features), one per fold —
                     raw per-fold SHAP values
    shap_imp_cv_df : DataFrame (n_folds, n_features) — mean |SHAP| per fold;
                     columns = feature names
    shap_stat_cv_df : DataFrame (7, n_features) from mean_sd_se_tval_pval_ci
    X_trn_cv       : list of ndarray (n_test_i, n_features), one per fold —
                     scaled input features (body-transformed) at test time
    """
    print("\n━━━  SHAP Analysis (LinearExplainer, out-of-fold)  ━━━")
    shap_cv = []
    X_trn_cv    = []

    for fold, ((_, te), estimator) in enumerate(zip(cv.split(X, y), estimators), 1):
        if isinstance(estimator, Pipeline):
            body, head = pipeline_behead(estimator)
            X_trn_te   = body.transform(X[te])
        else:
            X_trn_te = X[te]
            head     = estimator
        predictor = get_predictor(head)
        X_trn_cv.append(X_trn_te)
        masker    = shap.maskers.Independent(X_trn_te, max_samples=min(200, len(te)))
        explainer = shap.LinearExplainer(predictor, masker=masker)
        shap_cv.append(explainer.shap_values(X_trn_te))
        print(f"   fold {fold} done")

    shap_imp_cv_df = pd.DataFrame(
        #np.vstack([np.abs(imps).mean(axis=0) for imps in shap_cv]),
        np.vstack([shap.mean(axis=0) for shap in shap_cv]),
        columns=features
    )
    if normalise:
        shap_imp_cv_df = shap_imp_cv_df / shap_imp_cv_df.mean(axis=0).sum()

    shap_stat_cv_df = mean_sd_se_tval_pval_ci(shap_imp_cv_df)
    shap_stat_cv_df.loc["rank"] = shap_stat_cv_df.loc["pval"].rank(ascending=True)

    return shap_cv, shap_imp_cv_df, shap_stat_cv_df, X_trn_cv


def plot_shap_analysis(shap_cv: list,
                       X_trn_cv: list,
                       features: list,
                       cv,
                       X: np.ndarray,
                       y: np.ndarray) -> None:
    """
    Plot SHAP beeswarm, global bar chart, and dependence plots for the top-2
    features.

    Parameters
    ----------
    shap_cv : list of ndarray (n_test_i, n_features), one per fold — from shap_analysis
    X_trn_cv    : list of ndarray (n_test_i, n_features), one per fold — from shap_analysis
    features    : ordered list of feature names
    cv          : CV splitter used during fitting
    X           : raw feature matrix (used for OOF reconstruction)
    y           : response array (used for OOF reconstruction)
    """
    shap_vals = oof_arrays_from_cv(shap_cv, cv, X, y)
    X_trn     = oof_arrays_from_cv(X_trn_cv, cv, X, y)

    shap.summary_plot(shap_vals, X_trn, feature_names=features, show=False, plot_type="dot")
    plt.title("SHAP Beeswarm  —  log-odds contribution per patient (OOF)", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.savefig("shap_beeswarm.pdf", bbox_inches="tight")
    plt.show()
    print("✔  Saved shap_beeswarm.png")

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0),
                         index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(features))))
    mean_abs.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|  (log-odds units)", fontsize=11)
    ax.set_title("SHAP Global Importance — Lithium Response\n"
                 "Correlated features share credit — no double-counting", fontsize=12)
    fig.tight_layout()
    fig.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved shap_bar.png / .pdf")

    top2 = mean_abs.nlargest(min(2, len(features))).index.tolist()
    fig, axes = plt.subplots(1, len(top2), figsize=(6 * len(top2), 4), squeeze=False)
    for ax, feat in zip(axes[0], top2):
        idx = features.index(feat)
        ax.scatter(X_trn[:, idx], shap_vals[:, idx],
                   c=shap_vals[:, idx], cmap="coolwarm", alpha=0.6, s=20)
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xlabel(f"{feat}  (standardised)")
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP Dependence — {feat}", fontweight="bold")
    fig.tight_layout()
    fig.savefig("shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved shap_dependence.png")


# ------------------------------------------------------------------------------
# Permutation Importance (via cv_test)
# ------------------------------------------------------------------------------

def permutation_importance_analysis(estimators: list,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    features: list,
                                    cv,
                                    normalise: bool = False) -> tuple:
    """
    Permutation importance evaluated out-of-fold using pre-fitted CV estimators:
    for each test fold we permute features and measure the drop in balanced
    accuracy using the pipeline already fitted on that fold's training data.

    Parameters
    ----------
    estimators : list of fitted pipelines (one per CV fold)
    X          : raw (unscaled) feature matrix
    y          : binary response array
    features   : ordered list of feature names
    cv         : CV splitter used during fitting

    Returns
    -------
    imps_cv        : list of ndarray (n_repeats, n_features), one per fold —
                     raw per-repeat importances
    perm_cv_df : DataFrame (n_folds, n_features) — per-fold mean |perm
                     importance|; columns = feature names
    perm_stat_cv_df : DataFrame (7, n_features) from mean_sd_se_tval_pval_ci
    """
    print("\n━━━  Permutation Importance (per cv fold, pre-fitted estimators)  ━━━")
    scorer   = make_scorer(balanced_accuracy_score)
    perm_cv = []

    for fold, ((_, te), estimator) in enumerate(zip(cv.split(X, y), estimators), 1):
        if isinstance(estimator, Pipeline):
            body, head = pipeline_behead(estimator)
            X_trn_te   = body.transform(X[te])
        else:
            X_trn_te = X[te]
            head     = estimator
            
        res = permutation_importance(
            head, X_trn_te, y[te],
            n_repeats=30, random_state=RANDOM_STATE,
            scoring=scorer, n_jobs=-1,
        )
        perm_cv.append(res.importances.T)
        print(f"   fold {fold} done")

    perm_cv_df = pd.DataFrame(
        #np.vstack([np.abs(imps).mean(axis=0) for imps in perm_cv]),
        np.vstack([imps.mean(axis=0) for imps in perm_cv]),
        columns=features
    )

    if normalise:
        perm_cv_df = perm_cv_df / perm_cv_df.mean(axis=0).sum()
        
    perm_stat_cv_df = mean_sd_se_tval_pval_ci(perm_cv_df)
    perm_stat_cv_df.loc["rank"] = perm_stat_cv_df.loc["pval"].rank(ascending=True)

    print(perm_stat_cv_df.round(4))
    return perm_cv, perm_cv_df, perm_stat_cv_df


# ------------------------------------------------------------------------------
# Bootstrap Coefficient Stability
# ------------------------------------------------------------------------------

def bootstrap_stability(estimator,
                        X: np.ndarray,
                        y: np.ndarray,
                        features: list,
                        B: int = 200,
                        normalise: bool = False) -> tuple:
    """
    Bootstrap coefficient stability by cloning and refitting the estimator on
    B resampled datasets, then extracting coefficients each time.

    If estimator is a Pipeline, pipeline_behead splits it into preprocessing
    body and prediction head; get_predictor unwraps any meta-estimator wrapper
    (e.g. GridSearchCV); get_coef extracts coef_ or feature_importances_.

    Parameters
    ----------
    estimator  : single unfitted sklearn estimator or Pipeline (cloned each resample)
    X          : raw feature matrix (any preprocessing lives inside estimator)
    y          : binary response array
    features   : ordered list of feature names
    B          : number of bootstrap resamples (default 200)

    Returns
    -------
    coefs_boot      : list of B ndarray (1, n_features) — raw per-resample coefs
    boot_imp_cv_df  : DataFrame (B, n_features) — |coef| per resample;
                      columns = feature names
    boot_stat_cv_df : DataFrame (7, n_features) from mean_sd_se_tval_pval_ci
    """
    print(f"\n━━━  Bootstrap Coefficient Stability  (B={B})  ━━━")
    rng        = np.random.default_rng(RANDOM_STATE)
    coefs_boot = []

    for _ in range(B):
        idx    = rng.choice(len(X), size=len(X), replace=True)
        fitted = clone(estimator).fit(X[idx], y[idx])

        if isinstance(fitted, Pipeline):
            _, head = pipeline_behead(fitted)
        else:
            head = fitted

        coefs_boot.append(get_coef(get_predictor(head)).reshape(1, -1))

    #boot_imp_cv_df  = pd.DataFrame(np.abs(np.vstack(coefs_boot)), columns=features)
    boot_imp_cv_df  = pd.DataFrame(np.vstack(coefs_boot), columns=features)
    
    if normalise:
        boot_imp_cv_df = boot_imp_cv_df / boot_imp_cv_df.mean(axis=0).sum()
    
    se = boot_imp_cv_df.std(axis=0)
    boot_stat_cv_df = mean_sd_se_tval_pval_ci(boot_imp_cv_df, se=se)
    boot_stat_cv_df.loc["rank"] = boot_stat_cv_df.loc["pval"].rank(ascending=True)

    print("\n  Bootstrap stability summary:")
    print(boot_stat_cv_df.round(4))
    return coefs_boot, boot_imp_cv_df, boot_stat_cv_df

# %%
# ------------------------------------------------------------------------------
# Individual Feature importance Analysis
# ------------------------------------------------------------------------------

from utils.sklearn_utils import Ensure2D
from sklearn import metrics
import sklearn.linear_model as lm

def single_feature_classif(X_train, X_test, y_train, y_test):
    
    make2d = Ensure2D()
    lr = lm.LogisticRegression(fit_intercept=True, class_weight='balanced')
    aucs = np.array([metrics.roc_auc_score(y_test,
                    lr.fit(make2d.transform(X_train[:, j]), y_train).decision_function(make2d.transform(X_test[:, j]))) 
                    for j in range(X_train.shape[1])])
    return aucs

# estimator = cv_results['estimator']
# cv = cv_test
# normalise =False

def univ_classif_feature_importance(estimators: list,
                             X: np.ndarray,
                             y: np.ndarray,
                             features: list,
                             cv,
                             normalise: bool = False) -> tuple:
    auc_univ_cv = []

    for fold, ((tr, te), estimator) in enumerate(zip(cv.split(X, y), estimators), 1):
        if isinstance(estimator, Pipeline):
            body, head = pipeline_behead(estimator)
            X_trn_tr   = body.transform(X[tr])
            X_trn_te   = body.transform(X[te])
        else:
            X_trn_tr = X[tr]
            X_trn_te = X[te]
            head     = estimator
        predictor = get_predictor(head)
        auc_univ_cv.append(single_feature_classif(X_trn_tr, X_trn_te, y[tr], y[te]))

    auc_univ_cv_df = pd.DataFrame(np.vstack(auc_univ_cv), columns=features)

    if normalise:
        auc_univ_cv_df = auc_univ_cv_df / auc_univ_cv_df.mean(axis=0).sum()

    auc_univ_stat_cv_df = mean_sd_se_tval_pval_ci(auc_univ_cv_df, m0=0.5, alternative='greater')
    auc_univ_stat_cv_df.loc["rank"] = auc_univ_stat_cv_df.loc["pval"].rank(ascending=True)

    print("\n AUC univ summary:")
    print(auc_univ_stat_cv_df.round(4))
    
    return auc_univ_cv, auc_univ_cv_df, auc_univ_stat_cv_df


# %%
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
from config import config

# %%
# ------------------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------------------

INPUT_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'

data = pd.read_csv(INPUT_DATA)

# Select Input = dataframe - (target + drop + residualization)
feature_columns = [c for c in data.columns if c not in [config['target']] + \
    config['drop'] + config['residualization']]

X = data[feature_columns].values
y = data[config['target']].map(config['target_remap'])

# Multiply CSF columns by -1
csf_indices = np.array([i for i, col in enumerate(feature_columns) if 'CSF' in col])
X[:, (csf_indices)] *= -1

assert X.shape[1] == len(feature_columns)  == 268 # Check that the number of columns is correct

# %%
# ------------------------------------------------------------------------------
# Residualization
# ------------------------------------------------------------------------------

# import importlib
# import ml_utils
# importlib.reload(ml_utils)

if config['residualization']:
    from mulm.residualizer import Residualizer, ResidualizerEstimator

    residualization_columns = config['residualization']
    residualization_formula = "+".join(residualization_columns)
    residualizer = Residualizer(data=data, formula_res=residualization_formula)

    # Extract design matrix and pack it with X
    Z = residualizer.get_design_mat(data=data)
    residualizer_estimator = ResidualizerEstimator(residualizer)
        
    # Pack Z with X
    ZX = residualizer_estimator.pack(Z, X)
    Z_, X_ = residualizer_estimator.upack(ZX)
    assert np.all(X_ == X)
    assert Z.shape[1] + len(feature_columns) == ZX.shape[1]  # Check that the number of columns is correct

    # Finally, we will use ZX as X for the models, and Z as the residualization part
    X = ZX

# %%
# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score, roc_auc_score
from ml_utils import group_by_roi
roi_groups = group_by_roi(feature_columns)
# Convert roi_groups to indices
roi_groups = {roi:[feature_columns.index(x) for x in cols] for roi, cols in roi_groups.items()}
roi_names = list(roi_groups.keys())

# Models
from ml_utils import make_models
from config import config, cv_val
from ml_utils import PredefinedSplit


cv_test = PredefinedSplit(json_file=config['cv_test'])


models = make_models(n_jobs_grid_search=5, cv_val=cv_val,
                    residualization_formula=residualization_formula,
                    residualizer_estimator=residualizer_estimator,
                    roi_groups=roi_groups)

model = clone(models['model-grpRoiLda+lrl2_resid-age+sex+site'])

# %%
# ------------------------------------------------------------------------------
# Fit model with nested CV and evaluate out-of-fold performance
# ------------------------------------------------------------------------------

cv_results = cross_validate(
    clone(model), X, y, cv=cv_test,
    scoring=["balanced_accuracy", "roc_auc"],
    return_train_score=True, return_estimator=True, n_jobs=5,
)

metrics_df = classification_report_cv(X, y, cv_results['estimator'], cv_test)
print(metrics_df)
"""
              metric  avg_folds  std_folds  se_folds    pooled  pval_pooled  pval_fold                                                                                             fold_values
0  Balanced Accuracy   0.703175   0.045910  0.020532  0.703174     0.000008   0.000450                 [0.7658730158730158, 0.675, 0.7083333333333333, 0.7333333333333334, 0.6333333333333333]
1            ROC-AUC   0.690185   0.072699  0.032512  0.693275     0.999749   0.003187    [0.7777777777777778, 0.6666666666666666, 0.7250000000000001, 0.7185185185185186, 0.5629629629629629]
2          Precision   0.565501   0.065887  0.029466  0.561404          NaN        NaN                                  [0.6153846153846154, 0.5, 0.5454545454545454, 0.6666666666666666, 0.5]
3   Recall (class 0)   0.661905   0.073648  0.032936  0.662162     0.003542   0.005861                                                 [0.6428571428571429, 0.6, 0.6666666666666666, 0.8, 0.6]
4   Recall (class 1)   0.744444   0.081271  0.036345  0.744186     0.000957   0.001923                                [0.8888888888888888, 0.75, 0.75, 0.6666666666666666, 0.6666666666666666]
5                 F1   0.639389   0.054219  0.024248  0.640000          NaN        NaN                    [0.7272727272727273, 0.6, 0.631578947368421, 0.6666666666666666, 0.5714285714285714]
6                MCC   0.395871   0.093937  0.042010  0.391954          NaN        NaN
"""

# %%
# ------------------------------------------------------------------------------
# Feature importance Analysis: 1 Compute importance
# ------------------------------------------------------------------------------


RANDOM_STATE = 8
PALETTE      = "coolwarm"
C_GRID       = 10.0 ** np.arange(-3, 1)                       # [1e-3, 1e-2, 1e-1, 1]
features = roi_names

## DEBUG
estimators = cv_results['estimator']
cv = cv_test
## DEBUG

# SHAP analysis
shap_cv, shap_imp_cv_df, shap_stat_cv_df, X_trn_cv = shap_analysis(cv_results['estimator'], X, y, features, cv_test, normalise=True)
shap_imp_cv_df.mean().describe()  # Check that importances sum to 1 when normalised

plot_shap_analysis(shap_cv, X_trn_cv, features, cv_test, X, y)
shap_imp_cv_df = shap_imp_cv_df[shap_stat_cv_df.loc["mean"].sort_values(ascending=False).index]

scale_factor = 1.0
n_features = shap_imp_cv_df.shape[1]
fig, ax = plt.subplots(figsize=(8 * scale_factor, max(4, n_features * 0.3) * scale_factor))
sns.barplot(data=shap_imp_cv_df, orient="h", palette="Reds_r", capsize=.1, errorbar="se", ax=ax)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("SHAP importance averaged within cv folds")
ax.set_xlabel("Mean absolute SHAP value")
plt.tight_layout()
plt.savefig("reports/importance_SHAP.png", dpi=150, bbox_inches="tight")
plt.show()

# Permutation importance analysis
perm_cv, perm_cv_df, perm_stat_cv_df = permutation_importance_analysis(cv_results['estimator'], X, y, features, cv_test, normalise=True)
perm_cv_df.mean().describe()

perm_cv_df = perm_cv_df[perm_stat_cv_df.loc["mean"].sort_values(ascending=False).index]
n_features = perm_cv_df.shape[1]
fig, ax = plt.subplots(figsize=(8 * scale_factor, max(4, n_features * 0.3) * scale_factor))
sns.barplot(data=perm_cv_df, orient="h", palette="Reds_r", capsize=.1, errorbar="se", ax=ax)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("Permutation importance averaged within cv folds")
ax.set_xlabel("Mean drop in balanced accuracy when permuted")
plt.tight_layout()
plt.savefig("reports/importance_permutation.png", dpi=150, bbox_inches="tight")
plt.show()

# Haufe forward model
haufe_cv, haufe_cv_df, haufe_stat_cv_df = haufe_feature_importance(cv_results['estimator'], X, y, features, cv_test, normalise=True)

haufe_cv_df = haufe_cv_df[haufe_stat_cv_df.loc["mean"].sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8 * scale_factor, max(4, n_features * 0.3) * scale_factor))
sns.barplot(data=haufe_cv_df, orient="h", palette="Reds_r", capsize=.1, errorbar="se", ax=ax)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("Haufe |forward pattern| averaged within cv folds")
ax.set_xlabel("Haufe forward (Cov(X, s))")
plt.tight_layout()
plt.savefig("reports/importance_haufe.png", dpi=150, bbox_inches="tight")
plt.show()

# Bootstrap stability
coefs_boot, boot_imp_cv_df, boot_stat_cv_df = bootstrap_stability(clone(model), X, y, features, normalise=True)

boot_imp_cv_df = boot_imp_cv_df[boot_stat_cv_df.loc["mean"].sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8 * scale_factor, max(4, n_features * 0.3) * scale_factor))
sns.barplot(data=boot_imp_cv_df, orient="h", palette="Reds_r", capsize=.1, errorbar="se", ax=ax)
#sns.boxplot(data=boot_imp_cv_df, orient="h", palette="Reds_r", ax=ax)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title(f"Bootstrap |coef| Stability  (B={boot_imp_cv_df.shape[0]} resamples)")
ax.set_xlabel("|Bootstrap coefficient| (standardised)")
plt.tight_layout()
plt.savefig("reports/importance_bootstrap.png", dpi=150, bbox_inches="tight")
plt.show()

# Individual feature importance
auc_univ_cv, auc_univ_cv_df, auc_univ_stat_cv_df = \
    univ_classif_feature_importance(estimators, X, y, features, cv_test, normalise=False)
auc_univ_cv_df = auc_univ_cv_df[auc_univ_stat_cv_df.loc["mean"].sort_values(ascending=False).index]
print(auc_univ_stat_cv_df.T.sort_values("pval", ascending=True).head(30))

fig, ax = plt.subplots(figsize=(8 * scale_factor, max(4, n_features * 0.3) * scale_factor))
sns.barplot(data=auc_univ_cv_df, orient="h", palette="Reds_r", capsize=.1, errorbar="se", ax=ax)
#sns.boxplot(data=boot_imp_cv_df, orient="h", palette="Reds_r", ax=ax)
ax.axvline(0.5, color="black", lw=0.8, linestyle="--")
ax.set_title("AUC with single feature averaged within cv folds +-SE")
ax.set_xlabel("--------")
plt.tight_layout()
plt.savefig("reports/importance_auc_univ.png", dpi=150, bbox_inches="tight")
plt.show()

# mannwhitneyu_4auc instead of one-sample t-test
from utils.stats_utils import mannwhitneyu_4auc
aucs = auc_univ_stat_cv_df.loc["mean"]
n_neg, n_pos = np.sum(y == 0), np.sum(y == 0)
#U, z, pval = mannwhitneyu_4auc(aucs, n_pos, n_neg, alternative= "greater")
U, z, pval = mannwhitneyu_4auc(aucs, n_pos, n_neg, alternative= 'two-sided')
# mannwhitneyu_4auc(0.64, n_pos, n_neg, alternative= "greater")
_, pval_corr, _, _ = multipletests(pval, method="fdr_bh")

auc_univ_stat_cv_df.loc["pval"] = pval_corr
auc_univ_stat_cv_df.loc["pval"].describe()
auc_univ_stat_cv_df.T.round(4)



# ──────────────────────────────────────────────────────────────────────────────
# Retrieve feature importance significance from permutation randomization test
# Two sheets are loaded:
#   "model-grpRoiLda+lrl2_..._forwd"   → Haufe forward pattern  (tval, tval_pval_rnd)
#   "model-grpRoiLda+lrl2_..._featur"  → Individual AUC         (mean AUC, tval_pval_rnd)
# Significance thresholding (< 0.05) is applied after loading.
# ------------------------------------------------------------------------------

_ROI_MAPPING_CSV = ("data/processed/roi-cat12vbm/"
                    "study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_mapping-v-4-to-v-5.csv")
_RANDOMIZE_XLSX  = "reports/40_randomize_feature_importance_v-20250715.xlsx"

def load_randomize_importance(sheet_name, value_cols, roi_mapping_csv=_ROI_MAPPING_CSV,
                               xlsx=_RANDOMIZE_XLSX):
    """Load a randomization-importance sheet and remap feature names to Neuromorphometrics ROI names.

    Parameters
    ----------
    sheet_name    : str   — sheet to read from the Excel file
    value_cols    : list  — columns to keep besides 'feature' (e.g. ["tval", "tval_pval_rnd"])
    roi_mapping_csv : str — path to the roi_src → roi_dst mapping CSV
    xlsx          : str   — path to the randomization Excel file

    Returns
    -------
    DataFrame with columns: Feature (roi_dst), roi_src, *value_cols
    """
    df = pd.read_excel(xlsx, sheet_name=sheet_name, usecols=["feature"] + value_cols)

    roi_map = (
        pd.read_csv(roi_mapping_csv)
        [["roi_src", "roi_dst"]]
        .drop_duplicates("roi_src")
        .set_index("roi_src")["roi_dst"]
    )

    return (
        df.assign(roi_src=df["feature"], Feature=df["feature"].map(roi_map))
        [["Feature", "roi_src"] + value_cols]
        .reset_index(drop=True)
    )


# Haufe forward pattern significance (pval threshold applied outside)
haufe_rnd_df = load_randomize_importance(
    sheet_name="model-grpRoiLda+lrl2_..._forwd",
    value_cols=["tval", "tval_pval_rnd"],
)
haufe_stat_cv_randomize_df = haufe_rnd_df[haufe_rnd_df["tval_pval_rnd"] < 0.05].reset_index(drop=True)
print(haufe_stat_cv_randomize_df)

"""
                      Feature                                roi_src      tval  tval_pval_rnd
0    Medial Postcentral Gyrus  MPoG postcentral gyrus medial segment  3.313397          0.025
1  Inferior Lateral Ventricle                           Inf Lat Vent  2.857070          0.027
2                    Amygdala                               Amygdala  3.449307          0.034
3                 Hippocampus                            Hippocampus  2.807440          0.045
"""

# Feature importance significance (mean + pval from randomization)
univ_auc_rnd_df = load_randomize_importance(
    sheet_name="model-grpRoiLda+lrl2_..._featur",
    value_cols=["mean", "tval", "tval_pval_rnd"],
)
feat_rnd_df =univ_auc_rnd_df.sort_values("tval_pval_rnd")
print(feat_rnd_df.round(4).head(10))
"""
                       Feature                                             roi_src    mean    tval  tval_pval_rnd
1             Fourth Ventricle                                       4th Ventricle  0.6521  5.7928          0.004
0       Inferior Frontal Gyrus  OpIFG opercular part of the inferior frontal gyrus  0.6516  6.0834          0.007
2            Frontal Operculum                                FO frontal operculum  0.6373  4.8610          0.011
5            Lateral Ventricle                                   Lateral Ventricle  0.6070  4.1420          0.012
4       Calcarine and Cerebrum                               Calc calcarine cortex  0.5776  4.1805          0.014
3             Subcallosal Area                                SCA subcallosal area  0.6084  4.3865          0.016
7   Inferior Lateral Ventricle                                        Inf Lat Vent  0.5958  3.9811          0.029
8        Lateral Orbital Gyrus                          LOrG lateral orbital gyrus  0.6057  3.8730          0.034
10                 Hippocampus                                         Hippocampus  0.6689  2.9636          0.048
14                     Putamen                                             Putamen  0.5908  2.6181          0.050
"""


# %%
# ------------------------------------------------------------------------------
# Plot top ranked features
# ------------------------------------------------------------------------------

# Data organisation

all_stat_df = pd.concat([
    df.T.rename_axis("Feature").reset_index().assign(Method=method).sort_values("pval")
    for method, df in [
        ("SHAP",        shap_stat_cv_df),
        ("Permutation", perm_stat_cv_df),
        ("Haufe",       haufe_stat_cv_df),
        ("Individual",  auc_univ_stat_cv_df),
        #("Bootstrap",   boot_stat_cv_df),
    ]]
, ignore_index=True)
all_stat_df['log_pval']      = -np.log10(all_stat_df["pval"])
all_stat_df['log_pval_corr'] = -np.log10(all_stat_df["pval_corr"])


# Distribution of -log p-values
ax = sns.violinplot(data=all_stat_df, x="Method", y="log_pval")
ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
all_stat_df.groupby("Method")['log_pval'].describe()
all_stat_df.groupby("Method")['pval'].describe()


# Select % top best ranked features
methods_used_to_rank = ["SHAP", "Permutation", "Haufe"]
feature_order = (
    all_stat_df[all_stat_df.Method.isin(methods_used_to_rank)].groupby("Feature")["log_pval"].median()
    .sort_values(ascending=False)#.index.tolist()
)

# Haufe bars: star for randomization-significant features
haufe_annot_df = (
    haufe_stat_cv_randomize_df[["Feature"]]
    .assign(Method="Haufe", text="*")
)

sig_threshold = -np.log10(0.05)

univ_sig_df = (
    all_stat_df[all_stat_df["Method"] == "Individual"]
    .query("log_pval > @sig_threshold")
    .merge(univ_auc_rnd_df[["Feature", "mean"]].rename(columns={"mean": "auc_mean"}),
           on="Feature", how="inner")
    .assign(text=lambda d: d["auc_mean"].apply(lambda v: f"AUC={v:.2f}"),
            Method="Individual")
    [["Feature", "Method", "text"]]
)

# %% Bar plot: -log10(pval) per feature, colored by Method

toppercent = 100 / 100
features_top = feature_order[:int(np.ceil(toppercent * len(feature_order)))]
print(features_top)

all_stat_selected_df = all_stat_df[all_stat_df["Feature"].isin(features_top.index)]


n_f = len(all_stat_selected_df["Feature"].unique())
_height = 10
# Scale annotation fontsize to available vertical space per feature row
text_fontsize = max(9, int(72 * _height / n_f * 0.22))   # ~22% of row height in pt
annot_fontsize = text_fontsize + 2
label_fontsize = text_fontsize + 2
tick_fontsize  = text_fontsize + 2
star_fontsize  = text_fontsize + 8
palette = {"Haufe": "#1f77b4", "Permutation": "#ff7f0e",
            "SHAP": "#2ca02c", "Individual": "#7f7f7f"}

fig, ax = plt.subplots(figsize=(8, max(4, n_f * 0.4)))

g = sns.catplot(
    data=all_stat_selected_df, kind="bar",
    y="Feature", x="log_pval", hue="Method",
    order=features_top.index,
    orient="h",
    height=_height,
    aspect=1.0,
    palette=palette,
    #width=0.7,
    #gap=1.15
)

# Add hatching to Individual bars and its legend patch
import matplotlib.colors as mcolors
univ_rgb = mcolors.to_rgb("#7f7f7f")
for patch in g.ax.patches:
    if np.allclose(patch.get_facecolor()[:3], univ_rgb, atol=0.01):
        patch.set_hatch("//")
        patch.set_edgecolor("white")
for patch in g.legend.get_patches():
    if np.allclose(patch.get_facecolor()[:3], univ_rgb, atol=0.01):
        patch.set_hatch("//")
        patch.set_edgecolor("white")

import matplotlib.colors as mcolors

def annotate_bars(ax, annot_df, feat_order, method_order,
                  offset=0.02, fontsize=10, **text_kws):
    """Annotate the tip of each bar identified by (Feature, Method).

    Reads x (bar tip) and y (bar center) directly from patch geometry.
    method_order must match the top-to-bottom display order of bars within
    each feature group (same as the legend order in seaborn catplot).

    Parameters
    ----------
    ax           : matplotlib Axes from the catplot
    annot_df     : DataFrame with columns Feature, Method, text
    feat_order   : list of feature names in display order (top → bottom)
    method_order : list of method names in within-group order (top → bottom)
    offset       : horizontal gap beyond bar tip
    fontsize     : font size
    **text_kws   : forwarded to ax.text()
    """
    n_methods = len(method_order)

    # Feature → category y-center from tick labels; fallback to integer index
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    tick_pos    = ax.get_yticks()
    feat_ypos   = {lbl: pos for lbl, pos in zip(tick_labels, tick_pos) if lbl}
    if not feat_ypos:
        feat_ypos = {feat: i for i, feat in enumerate(feat_order)}

    # Group patches by nearest feature category, then sort within each group
    feat_bars = {feat: [] for feat in feat_order}
    for p in ax.patches:
        if p.get_height() <= 0:
            continue
        y_center = p.get_y() + p.get_height() / 2
        feat = min(feat_ypos, key=lambda f: abs(feat_ypos[f] - y_center))
        feat_bars[feat].append((y_center, p.get_x() + p.get_width()))

    # Within each feature, ascending y = top→bottom = method_order
    bar_map = {}
    bar_heights = []
    for feat in feat_order:
        bars = sorted(feat_bars[feat], key=lambda b: b[0])[:n_methods]
        for method_idx, (y_center, x_tip) in enumerate(bars):
            bar_map[(feat, method_order[method_idx])] = (x_tip, y_center)
        bar_heights.extend(h for _, h in feat_bars[feat])

    # Small downward shift so text visual center aligns with bar center
    # (in seaborn's inverted y-axis, positive dy moves the label downward)
    bar_h = np.mean([p.get_height() for p in ax.patches if p.get_height() > 0])
    dy = bar_h * 0.25

    kws = dict(va="center", ha="left", clip_on=False)
    kws.update(text_kws)
    n_placed = 0
    for _, row in annot_df.iterrows():
        key = (row["Feature"], row["Method"])
        if key not in bar_map:
            continue
        x_tip, y_center = bar_map[key]
        ax.text(x_tip + offset, y_center + dy, str(row["text"]), fontsize=fontsize, **kws)
        n_placed += 1
    print(f"annotate_bars: placed {n_placed}/{len(annot_df)} annotations")


_feat_order   = list(features_top.index)
# Method order top→bottom within each feature group (matches legend order)
_method_order = [t.get_text() for t in g.legend.get_texts()]

# Haufe bars: star for randomization-significant features
annotate_bars(g.ax, haufe_annot_df, _feat_order, _method_order,
              fontsize=star_fontsize, color=palette["Haufe"], fontweight="bold")

# Individual bars: AUC label for features with log_pval > threshold
annotate_bars(g.ax, univ_sig_df, _feat_order, _method_order,
              fontsize=annot_fontsize, color=palette["Individual"])


g.ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=2)
g.set_axis_labels("-log10(p-value)", "ROI", fontsize=label_fontsize)
g.ax.tick_params(axis="both", labelsize=tick_fontsize)
g.legend.set_title(g.legend.get_title().get_text(), prop={"size": tick_fontsize})
for t in g.legend.get_texts(): t.set_fontsize(tick_fontsize)
g.tight_layout()
g.savefig("reports/importance_roi_logpval_barplot.png", dpi=150, bbox_inches="tight")
g.savefig("reports/importance_roi_logpval_barplot.svg", bbox_inches="tight")
g.savefig("reports/importance_roi_logpval_barplot.pdf", bbox_inches="tight")

# %%
# Agrement between Methods

# Rank table: features (rows) x methods (columns), ranked by -log10(pval)
rank_df = (
    all_stat_df
    .assign(rank=lambda d: d.groupby("Method")["log_pval"]
                            .rank(ascending=False).astype(int))
    .pivot(index="Feature", columns="Method", values="rank")
    .reindex(feature_order.index)
)
print(rank_df.head(20))

# Methods correlation matrix

# Correlation of log_pval between Methods (Features as rows)
from utils.eda import plot_correlation
logpval_methods_df = all_stat_df.pivot(index="Feature", columns="Method", values="log_pval")
plot_correlation(logpval_methods_df, corr="spearman", annot=True, linewidths=0.5)
plot_correlation(logpval_methods_df, corr="pearson", annot=True, linewidths=0.5)
# => Rankin methods a poorly correlated


# %% Save results to Excel
excel_path = "reports/importance_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    metrics_df.to_excel(writer,            sheet_name="global classif. metrics")
    shap_stat_cv_df.T.sort_values("pval", ascending=True).to_excel(writer,       sheet_name="shap_importance")
    perm_stat_cv_df.T.sort_values("pval", ascending=True).to_excel(writer,       sheet_name="perm_importance")
    haufe_stat_cv_df.T.sort_values("pval", ascending=True).to_excel(writer,      sheet_name="haufe_importance")
    boot_stat_cv_df.T.sort_values("pval", ascending=True).to_excel(writer,       sheet_name="boot_stability")
    auc_univ_stat_cv_df.T.sort_values("pval", ascending=True).to_excel(writer,   sheet_name="individual_auc_importance")
    all_stat_df.to_excel(writer,           sheet_name="all_importance",       index=False)
print(f"✔  Saved {excel_path}")

# %%
# ------------------------------------------------------------------------------
# Final SHAP Analaysis using: Logistic Linear Regression with L2 regularization using only "Left Amygdala_GM_Vol", "Left Hippocampus_GM_Vol"


# %%
# ------------------------------------------------------------------------------
# Sub-analysis: Amygdala + Hippocampus only (left hemisphere, GM volume)
# ------------------------------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate as _cv

amy_hippo_gm_features = ["Left Amygdala_GM_Vol", "Left Hippocampus_GM_Vol"]

Amygdala =  ["Left Amygdala_GM_Vol", "Left Amygdala_CSF_Vol",
             "Right Amygdala_GM_Vol", "Right Amygdala_CSF_Vol"]

Hippocampus = ["Left Hippocampus_GM_Vol", "Left Hippocampus_CSF_Vol",
               "Right Hippocampus_GM_Vol", "Right Hippocampus_CSF_Vol"]

MiddleTemporalGyrus = ["Left Middle Temporal Gyrus_GM_Vol", "Left Middle Temporal Gyrus_CSF_Vol",
                       "Right Middle Temporal Gyrus_GM_Vol", "Right Middle Temporal Gyrus_CSF_Vol"]


features_selected = amy_hippo_gm_features
features_selected = Amygdala + Hippocampus# + MiddleTemporalGyrus


X = data[features_selected].values
#assert X.shape == (117, 2)


if config['residualization']:
    from mulm.residualizer import Residualizer, ResidualizerEstimator

    residualization_formula = "+".join(config['residualization'])
    residualizer = Residualizer(data=data, formula_res=residualization_formula)

    # Extract design matrix and pack it with X
    Z = residualizer.get_design_mat(data=data)
    residualizer_estimator = ResidualizerEstimator(residualizer)
        
    # Pack Z with X
    X = residualizer_estimator.pack(Z, X)
    #assert X.shape == (117, 20) # 18 residualization columns + 2 features


model_sel = Pipeline([
    ("residualizer", residualizer_estimator),
    ("scaler",       StandardScaler()),
    ("clf",          GridSearchCV(
        LogisticRegression(fit_intercept=False, class_weight="balanced"),
        {"C": 10. ** np.arange(-3, 1)},
        cv=cv_val, n_jobs=5, scoring="accuracy",
    )),
])

model_sel.fit(X, y)

cv_results = cross_validate(
    model_sel, X, y, cv=cv_test,
    scoring=["balanced_accuracy", "roc_auc"],
    return_train_score=True, return_estimator=True, n_jobs=5,
)

# SHAP analysis
shap_cv, shap_imp_cv_df, shap_stat_cv_df, X_trn_cv = shap_analysis(cv_results['estimator'], 
                                                                   X, y, features_selected, cv_test, normalise=True)
shap_imp_cv_df.mean().describe()  # Check that importances sum to 1 when normalised
plot_shap_analysis(shap_cv, X_trn_cv, features_selected, cv_test, X, y)


print(shap_stat_cv_df.round(4))