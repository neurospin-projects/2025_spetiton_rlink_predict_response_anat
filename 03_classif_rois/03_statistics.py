"""
Univariate statistics
"""


import pandas as pd
import numpy as np
import os

from utils.stats_pairwise import pairwise_stats
from config import config
import seaborn as sns
from utils.pandas_utils import merge_on_index


#ROOT = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup/"
ROOT = "/home/ed203246/git/2025_spetiton_rlink_predict_response_anat/03_classif_rois/"
ATLAS_MAPPING_FILE = "./data/lobes_Neuromorphometrics_with_dfROI_correspondencies.csv"
#INPUT_CAT12_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'
INPUT_CAT12_DATA = './data/processed/roi-cat12vbm/df_ROI-notscaled_age_sex_site_M00_v4labels.csv'
INPUT_FS_DATA = './data/processed/fs_aseg_volumes.tsv'
PARTICIPANTS_DATA= './data/participants.tsv'
RESPONSE_DATA= './data/dataset-outcome_version-4.tsv'
# VBM
FEATURES_HIPAMY_VBM = ['Left Hippocampus_GM_Vol', 'Right Hippocampus_GM_Vol',
                'Left Amygdala_GM_Vol', 'Right Amygdala_GM_Vol']
# FreeSurfer
FEATURES_HIPAMY_FS = ['Left-Hippocampus', 'Right-Hippocampus',
                'Left-Amygdala', 'Right-Amygdala']


OUTPUT = "reports/classification_fs_reports.xlsx"


# M3-M0 statistics / plot
# inputs
VOL_FILE_VBM = "./data/processed/labels_Neuromorphometrics.nii"
VBMLOOKUP_FILE = "./data/processed/labels_Neuromorphometrics.xml"
DATA_DIR = ROOT+"data/processed/"
#M0_DATA = DATA_DIR+"df_ROI_age_sex_site_M00_v4labels.csv"
M3_MINUS_M0_DATA = DATA_DIR+"/df_ROI_M03_minus_M00_age_sex_site_v4labels.csv"
M0M3_DATA=DATA_DIR+"df_ROI_age_sex_site_M00_M03_v4labels.csv"


def load_response():
    response = pd.read_csv(RESPONSE_DATA, index_col="participant_id", sep='\t')
    label = 'Response.Status.at.end.of.follow.up'
    response = response[[label]]
    response = response.rename(columns={label:"response"})
    response = response[response['response'].isin(["NR","PaR","GR"])]
    return response


def load_fs_data(features = []):
    data_fs = pd.read_csv(INPUT_FS_DATA, index_col="participant_id", sep="\t")
    data_fs = data_fs[data_fs.session == "M00"]
    data_fs.drop(columns=["session"], inplace=True)
    data_fs = data_fs
    
    # age, sex from participants
    participants = pd.read_csv(PARTICIPANTS_DATA, index_col="participant_id", sep='\t')[["age", "sex", "ses-M00_center"]]
    participants = participants.rename(columns={"ses-M00_center":'site'})
    
    data_fs = pd.merge(data_fs, participants, left_index=True, right_index=True, how="left",)
    data_fs['site'] = data_fs['site'].astype(int).apply(lambda x: f"site-{x:02d}")
    
    # Response
    response = load_response()
    data_fs = pd.merge(data_fs, response, left_index=True, right_index=True, how="left",)
    assert data_fs.shape[0] == 135
    data_fs = data_fs[data_fs['response'].isin(["NR","PaR","GR"])]
    
    return data_fs


    # column of population dataframe that defines response to Li label

def diff_lists(l1, l2):
    s1, s2 = set(l1), set(l2)
    only_l1 = sorted(s1 - s2)
    only_l2 = sorted(s2 - s1)
    rows = (
        [{"value": v, "l1": True,  "l2": False} for v in only_l1] +
        [{"value": v, "l1": False, "l2": True}  for v in only_l2]
    )
    return pd.DataFrame(rows, columns=["value", "l1", "l2"])


def make_varname_lookup(names, rmsuffix=None):
    import re
    lookup = {}
    for name in names:
        safe = re.sub(r'[^a-zA-Z0-9]', '_', name)
        if safe and safe[0].isdigit():
            safe = '_' + safe
        if rmsuffix is not None:
            safe_suffix = re.sub(r'[^a-zA-Z0-9]', '_', rmsuffix)
            if safe.endswith(safe_suffix):
                safe = safe[: -len(safe_suffix)]
        lookup[name] = safe
    return lookup, {v: k for k, v in lookup.items()}


def lm(formula, data, return_stats=[]):
    import statsmodels.formula.api as sm
    model = sm.ols(formula=formula, data=data)
    lmfit = model.fit()
    # or all at once as a clean DataFrame
    stats = pd.DataFrame({
        "coef":    lmfit.params[return_stats],
        "t":       lmfit.tvalues[return_stats],
        "p":       lmfit.pvalues[return_stats],
    })
    return stats, lmfit

################################################################################
# %% Load Data
# ============

def load_data():
    merge_cols = ["age", "sex", "response"]
    participants = pd.read_csv(PARTICIPANTS_DATA, index_col="participant_id", sep='\t')[["age", "sex"]]

    # VBM baseline
    data_vbm_m0 = pd.read_csv(INPUT_CAT12_DATA, index_col="participant_id")
    #data_vbm_m0 = load_vbm_data()
    # data_vbm_m0["response"] = data_vbm_m0["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})
    assert data_vbm_m0.shape[0] == 117
    print(data_vbm_m0.groupby('sex')['tiv'].mean())
    # Check participant_id, age and sex from data_vbm_m0 exactly match participants
    assert merge_on_index(data_vbm_m0, participants, on= ["age", "sex"]).shape[0] == data_vbm_m0.shape[0]
    #data_vbm_m0.columns = data_vbm_m0.columns.str.replace(" ", "_")

    # VBM change
    data_vbm_change = pd.read_csv(M3_MINUS_M0_DATA, index_col="participant_id")
    data_vbm_change["sex"] = data_vbm_change["sex"].map({'female': 'F', 'male': 'M'})
    # Check participant_id, age and sex from data_vbm_m0 exactly match participants
    assert merge_on_index(data_vbm_change, participants, on= ["age", "sex"]).shape[0] == data_vbm_change.shape[0]

    # Only diff data_vbm_m0 has TIV
    len(diff_lists(data_vbm_m0.columns, data_vbm_change.columns)) == 1

    ## FS
    data_fs = load_fs_data()
    # data_fs["response"] = data_fs["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})

    #data_fs.columns = data_fs.columns.str.replace("-", "_")
    assert data_fs.shape[0] == 116

    global_features_fs = \
    ['BrainSeg', 'BrainSegNotVent', 'CerebralWhiteMatter', 'Cortex', 
    'EstimatedTotalIntraCranialVol', 'Mask', 'SubCortGray', 'SupraTentorial',
    'SupraTentorialNotVent', 'SurfaceHoles', 'TotalGray', 'VentricleChoroidVol', 
    'lhCerebralWhiteMatter', 'lhCortex', 'lhSurfaceHoles', 'rhCerebralWhiteMatter', 
    'rhCortex', 'rhSurfaceHoles']
    
    # Merge 
    data_vbm_fs = merge_on_index(data_vbm_m0[merge_cols + FEATURES_HIPAMY_VBM],
                                data_fs[merge_cols + FEATURES_HIPAMY_FS], on=merge_cols)
    assert data_vbm_fs.shape[0] == 116
    #import seaborn as sns

    return data_vbm_m0, data_vbm_change, data_fs, data_vbm_fs
# data_fs = pd.merge(data_fs, data_vbm_m0[["response", "age", "sex", "site"]],
#                    left_index=True, right_index=True, how='left')


def onesample_stats(data, cols=None):
    from scipy.stats import ttest_1samp, wilcoxon
    if cols is None:
        cols = data.columns
    rows = []
    for col in cols:
        x = data[col].dropna()
        t_stat, t_p = ttest_1samp(x, popmean=0)
        w_stat, w_p = wilcoxon(x)
        cohens_d = x.mean() / x.std(ddof=1)
        rows.append({"dvar": col,
                     "ivar":1,
                     "mean": x.mean(), "median": x.median(),
                     "t": t_stat, "t_p": t_p,
                     "wilcoxon": w_stat, "wilcoxon_p": w_p,
                     "cohens_d": cohens_d})
    return pd.DataFrame(rows)


def twosample_stats(data, grp, cols=None):
    from scipy.stats import ttest_ind, mannwhitneyu
    if cols is None:
        cols = [c for c in data.columns if c != grp]
    grp_vals = sorted(data[grp].dropna().unique())
    if len(grp_vals) != 2:
        raise ValueError(f"grp column must have exactly 2 unique values, got {grp_vals}")
    g0, g1 = grp_vals
    rows = []
    for col in cols:
        x0 = data.loc[data[grp] == g0, col].dropna()
        x1 = data.loc[data[grp] == g1, col].dropna()
        t_stat, t_p = ttest_ind(x0, x1, equal_var=False)
        u_stat, u_p = mannwhitneyu(x0, x1, alternative="two-sided")
        n0, n1 = len(x0), len(x1)
        pooled_std = np.sqrt(((n0 - 1) * x0.std(ddof=1) ** 2 + (n1 - 1) * x1.std(ddof=1) ** 2) / (n0 + n1 - 2))
        cohens_d = (x0.mean() - x1.mean()) / pooled_std
        rows.append({
            "dvar": col,
            "ivar": grp,
            f"mean_{g0}": x0.mean(), f"mean_{g1}": x1.mean(),
            f"median_{g0}": x0.median(), f"median_{g1}": x1.median(),
            "welch_t": t_stat, "welch_p": t_p,
            "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
            "cohens_d": cohens_d,
        })
    return pd.DataFrame(rows)


# %% Linear regression univariate statistics
# ==========================================

def stats_lm(data, features, formula, return_stats):
    stats_list = []
    for feat in features:
        stats, lmfit = lm(
            formula=formula % feat,
            data=data,
            return_stats=return_stats
        )
        stats["dvar"] = feat
        stats_list.append(stats)

    stats = pd.concat(stats_list)
    stats = stats.reset_index(names="ivar")
    print(stats)
    return stats


def multipletests_correct(stats, method="fdr_bh"):
    """Apply multipletests correction to columns whose name (or first MultiIndex level) is 'p', 'pval', 'p_*', or 'pval_*'.
    Adds new columns with the prefix replaced by 'p_corr' or 'pval_corr'.
    """
    from statsmodels.stats.multitest import multipletests

    _PREFIX_MAP = {"p": "p_corr", "pval": "pval_corr"}

    def _new_prefix(col0):
        for pfx, new_pfx in _PREFIX_MAP.items():
            if col0 == pfx or col0.startswith(pfx + "_"):
                return new_pfx + col0[len(pfx):]
        return None

    stats = stats.copy()
    if isinstance(stats.columns, pd.MultiIndex):
        p_cols = [c for c in stats.columns if _new_prefix(c[0]) is not None]
        for col in p_cols:
            _, p_corr, _, _ = multipletests(stats[col].fillna(1), method=method)
            new_col = (_new_prefix(col[0]),) + col[1:]
            stats[new_col] = p_corr
    else:
        p_cols = [c for c in stats.columns if _new_prefix(c) is not None]
        for col in p_cols:
            _, p_corr, _, _ = multipletests(stats[col].fillna(1), method=method)
            stats[_new_prefix(col)] = p_corr
    return stats


def coef_to_prop_change(stats, data, grp=None, grp_ref=None):
    """Transform coef* columns into proportion of change relative to the mean of data.

    For each column whose name (or first MultiIndex level) starts with 'coef',
    adds a new 'prop_change*' column = coef / mean(dvar) where mean(dvar) is
    computed either across all rows (grp=None) or within the reference group.

    Parameters
    ----------
    stats    : DataFrame with dvars as index and coef* columns
    data     : DataFrame whose columns include all values in stat.index
    grp      : column name in data used to select the reference group (optional)
    grp_ref  : value of data[grp] defining the reference group (optional)
    """
    def _new_name(col0):
        if col0 == "coef" or col0.startswith("coef_"):
            return "prop_change" + col0[len("coef"):]
        return None

    if grp is not None and grp_ref is not None:
        mean_dvars = data.loc[data[grp] == grp_ref, stats.index].mean()
    else:
        mean_dvars = data[stats.index].mean()

    stat = stats.copy()
    if isinstance(stats.columns, pd.MultiIndex):
        coef_cols = [c for c in stats.columns if _new_name(c[0]) is not None]
        for col in coef_cols:
            new_col = (_new_name(col[0]),) + col[1:]
            stats[new_col] = stats[col] / mean_dvars
    else:
        coef_cols = [c for c in stats.columns if _new_name(c) is not None]
        for col in coef_cols:
            stats[_new_name(col)] = stats[col] / mean_dvars
    return stats


def average_lr_rois(data, vars):
    """
    Average left and right hemisphere columns for each ROI name in `vars`.

    All columns whose name contains the var string are averaged into a single
    bilateral column named after the var.

    Parameters
    ----------
    data : DataFrame
    vars : list of ROI base names, e.g. ["Hippocampus", "Amygdala"]

    Returns
    -------
    DataFrame with one new column per var (left/right originals are kept).
    """
    df = data.copy()
    for var in vars:
        lr_cols = [c for c in df.columns if var in c]
        if not lr_cols:
            raise ValueError(f"No columns found containing '{var}'")
        df[var] = df[lr_cols].mean(axis=1)
    return df


def partial_residuals(data, dvars, formula, res_vars, verbose=True):
    """
    Fit OLS models and add partial-residual columns for each dependent variable.

    For each dvar, fits `formula % dvar` and computes:
      y_adj = intercept + sum(res_vars contributions) + residuals
    stripping out all other covariate effects.

    Parameters
    ----------
    data     : DataFrame
    dvars    : dependent variable column names, e.g. ["Hippocampus", "Amygdala"]
    formula  : model formula template with one %s placeholder,
               e.g. "%s ~ response + sex + age + site + tiv"
    res_vars : independent variable of interest whose contribution is kept,
               e.g. ["response", "sex"]. Provide the residuals adjusted of all
               other ivars of the model (defined by the foumula).

    Returns
    -------
    DataFrame with one new column per dvar named "{dvar}_adj".
    """
    import statsmodels.formula.api as smf

    df = data.copy()
    for dvar in dvars:
        fit = smf.ols(formula % dvar, data=df).fit()
        ivars = np.array(fit.model.exog_names)
        missing = [f for f in res_vars if not any(f in n for n in ivars)]
        assert not missing, f"res_vars not found in model terms: {missing} (available: {list(ivars)})"
        keep = np.array([
            "Intercept" in iv or any(f in iv for f in res_vars)
            for iv in ivars
        ])
        if verbose:
            print("partial_residuals:")
            print("   - ind. vars:", ivars)
            print("   - adj. vars:", ivars[~keep])
            print("   - res. vars:", ivars[keep])
        df[f"{dvar}_adj"] = (
            fit.model.exog[:, keep] @ fit.params.values[keep] + fit.resid.values
        )

    return df


def extract_adjusted(data, dvars, ivars):
    """
    Select and clean up the output of `partial_residuals`.

    Keeps only the `ivars` columns (independent factors, e.g. "response", "sex")
    and the `{dvar}_adj` columns, renaming the latter by stripping the "_adj" suffix.

    Parameters
    ----------
    data  : DataFrame (output of `partial_residuals`)
    dvars : dependent variable base names, e.g. ["Hippocampus", "Amygdala"]
    ivars : independent variable columns to keep, e.g. ["response", "sex"]

    Returns
    -------
    DataFrame with columns ivars + dvars (adjusted values, suffix removed).
    """
    adj_cols = [f"{dvar}_adj" for dvar in dvars]
    df = data[ivars + adj_cols].copy()
    df = df.rename(columns={f"{dvar}_adj": dvar for dvar in dvars})
    return df


_PAGE_W_IN = 6.3   # A4 usable width (210mm - 2x25mm margins) in inches

def violinplot_adjusted(data, dvars, x=None, hue=None,
                        split=True, gap=0.1, inner="quart", swarm=False,
                        hue_statannotations=False, annot_text_format="star",
                        figsize=None, fontsize=7, sharey=False, hline=None):
    """
    Violin plots of adjusted dependent variable columns side by side.

    Expects `dvars` columns to already exist in `data` (e.g. produced by
    `extract_adjusted`).

    Default layout: each subplot occupies one quarter of page width so that
    two subplots together fill half a page (suitable for 2-column journal layout).

    Parameters
    ----------
    data                : DataFrame
    dvars               : dependent variable column names, e.g. ["Hippocampus", "Amygdala"]
    x                   : column used as x-axis grouping, e.g. "response"
    hue                 : column used for color grouping, e.g. "sex"
    split               : draw half-violins for each hue level (requires hue)
    gap                 : gap between split violin halves
    inner               : interior representation ("quart", "box", "stick", None)
    swarm               : overlay individual data points as a swarm plot.
                          With split=True, dodge=True is used so each hue group's
                          points align roughly with their violin half (swarmplot has
                          no native split mode, so alignment is approximate).
    hue_statannotations : if True, annotate Mann-Whitney U p-values for each pair
                          of hue levels within every x category (requires `hue`).
    annot_text_format   : "star" (default) shows only stars; "star+stat" shows
                          stars and the U statistic, e.g. "** (U=312.0)".
    figsize             : (width, height) in inches. Defaults to quarter-page
                          width per subplot, height = width * 1.8.
    fontsize            : base font size in points (default 7 for publication).
    """
    from itertools import combinations
    import matplotlib.pyplot as plt

    subplot_w = _PAGE_W_IN / 4          # quarter page per subplot
    if figsize is None:
        figsize = (subplot_w * len(dvars), subplot_w * 1.8)

    with plt.rc_context({
        "font.size":        fontsize,
        "axes.titlesize":   fontsize + 1,
        "axes.labelsize":   fontsize,
        "xtick.labelsize":  fontsize,
        "ytick.labelsize":  fontsize,
        "legend.fontsize":  fontsize,
    }):
        fig, axes = plt.subplots(1, len(dvars), figsize=figsize, sharey=sharey)
        if len(dvars) == 1:
            axes = [axes]

        if hue_statannotations and hue is not None:
            from statannotations.Annotator import Annotator
            hue_vals = sorted(data[hue].unique())
            if x is not None:
                x_vals = sorted(data[x].unique())
                pairs = [
                    [(x_val, h1), (x_val, h2)]
                    for x_val in x_vals
                    for h1, h2 in combinations(hue_vals, 2)
                ]
                annot_params = dict(data=data, x=x, y=None, hue=hue)
            else:
                pairs = list(combinations(hue_vals, 2))
                annot_params = dict(data=data, x=hue, y=None)

        for ax, dvar in zip(axes, dvars):
            sns.violinplot(data=data, x=x, y=dvar, hue=hue, ax=ax,
                           split=split, gap=gap, inner=inner)
            if swarm:
                swarm_palette = {v: "grey" for v in sorted(data[hue].unique())} \
                                 if hue is not None else None
                swarm_kws = dict(data=data, x=x, y=dvar, hue=hue, ax=ax,
                                 dodge=hue is not None, size=3, legend=False)
                if swarm_palette is not None:
                    swarm_kws["palette"] = swarm_palette
                else:
                    swarm_kws["color"] = "grey"
                sns.swarmplot(**swarm_kws)
            if hue_statannotations and hue is not None:
                annotator = Annotator(ax, pairs, **{**annot_params, "y": dvar})
                if annot_text_format == "star+stat":
                    annotator.configure(test="Mann-Whitney", text_format="star", verbose=0)
                    annotator.apply_test()
                    def _pval_to_stars(p):
                        if p <= 0.0001: return "****"
                        if p <= 0.001:  return "***"
                        if p <= 0.01:   return "**"
                        if p <= 0.05:   return "*"
                        return "ns"
                    custom_texts = []
                    for ann in annotator.annotations:
                        p = ann.data.pvalue
                        s = ann.data.stat_value
                        u_str = f"{s:.0f}" if abs(s) >= 10 else f"{s:.1f}"
                        custom_texts.append(f"{_pval_to_stars(p)}\nU={u_str}")
                    annotator.set_custom_annotations(custom_texts)
                    annotator.annotate()
                else:
                    annotator.configure(test="Mann-Whitney",
                                        text_format=annot_text_format).apply_and_annotate()
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
            if hline is not None:
                ax.axhline(hline, color="black", linewidth=1.0, linestyle="-", zorder=3)
            ax.set_title(dvar)
            if sharey and ax is not axes[0]:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("Adjusted value")
            ax.set_xlabel(x if x is not None else "")

        fig.tight_layout()
    return fig, axes



# %%
def stats_m0():
    # Load data
    data_vbm_m0, data_vbm_change, data_fs_m0, data_vbm_fs = load_data()
    data_vbm_m0["response"] = data_vbm_m0["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})
    data_fs_m0["response"] = data_fs_m0["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})
    data_fs_m0 = data_fs_m0.rename(columns={"EstimatedTotalIntraCranialVol":"tiv"})

    # keeep only GM features + rename to be valid in formulae
    # VBM
    features_vbm = [c for c in data_vbm_m0.columns if c not in ["age", "sex", "response", "site"] and "GM" in c]    
    lookup_vbm, lookup_vbm_revert = make_varname_lookup(data_vbm_m0.columns)
    data_vbm_m0_ = data_vbm_m0.rename(columns=lookup_vbm)
    "response" in data_vbm_m0_.columns
    data_vbm_m0_["response"]
    features_vbm_ = [lookup_vbm.get(f, f) for f in features_vbm]
    FEATURES_HIPAMY_VBM_ = [lookup_vbm.get(f, f) for f in FEATURES_HIPAMY_VBM]
    
    # FS
    features_fs = [c for c in data_fs_m0.columns if c not in ["age", "sex", "response", "site"]]
    lookup_fs, lookup_fs_revert = make_varname_lookup(data_fs_m0.columns)
    data_fs_m0_ = data_fs_m0.rename(columns=lookup_fs)
    features_fs_ = [lookup_fs.get(f, f) for f in features_fs]
    FEATURES_HIPAMY_FS_ = [lookup_fs.get(f, f) for f in FEATURES_HIPAMY_FS]
    
    sns.violinplot(data=data_vbm_m0_[FEATURES_HIPAMY_VBM_ + ['sex', 'tiv']], x='sex', y='tiv')
    sns.violinplot(data=data_fs_m0_[FEATURES_HIPAMY_FS_ + ['sex', 'tiv']], x='sex', y='tiv')
    sns.pairplot(data_vbm_m0_[FEATURES_HIPAMY_VBM_ + ['sex', 'tiv', "response"]], hue="response")


    # Pairwise stats
    stats_vbm_m0_resp_tests = pairwise_stats(data_vbm_m0, vars1=['response'],
                        vars2=features_vbm,
                        cattest="prop_ztest")
    stats_vbm_m0_resp_tests = stats_vbm_m0_resp_tests.sort_values('pval')
    print("OUTPUT", stats_vbm_m0_resp_tests)
    
    
    # Response (Unscaled data, use tiv as regressor)
    stats_vbm_m0_resp_lm = stats_lm(data_vbm_m0_, features_vbm_,
                formula="%s ~ response + sex + age + site + tiv",
                return_stats=["response", "age", "sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_vbm_m0_resp_lm = multipletests_correct(stats_vbm_m0_resp_lm, method="fdr_bh")
    stats_vbm_m0_resp_lm = coef_to_prop_change(stats=stats_vbm_m0_resp_lm, data=data_vbm_m0_, grp='response', grp_ref=0)
    stats_vbm_m0_resp_lm["Modality"] = "VBM"
    stats_vbm_m0_resp_lm = stats_vbm_m0_resp_lm.sort_values(("p", "response"))
    print("OUTPUT", stats_vbm_m0_resp_lm)
 
    stats_fs_m0_resp_lm = stats_lm(data_fs_m0_, features_fs_,
                formula="%s ~ response + sex + age + site + tiv",
                return_stats=["response", "age", "sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_fs_m0_resp_lm["Modality"] = "FS"
    stats_fs_m0_resp_lm = stats_fs_m0_resp_lm.sort_values(("p", "response"))
    stats_fs_m0_resp_lm_hipamy = stats_fs_m0_resp_lm[stats_fs_m0_resp_lm.index.isin(FEATURES_HIPAMY_FS_)]
    stats_fs_m0_resp_lm = multipletests_correct(stats_fs_m0_resp_lm, method="fdr_bh")
    stats_fs_m0_resp_lm = coef_to_prop_change(stats=stats_fs_m0_resp_lm, data=data_fs_m0_, grp='response', grp_ref=0)
    stats_fs_m0_resp_lm_hipamy = multipletests_correct(stats_fs_m0_resp_lm_hipamy, method="fdr_bh")
    stats_fs_m0_resp_lm_hipamy = coef_to_prop_change(stats=stats_fs_m0_resp_lm_hipamy, data=data_fs_m0_, grp='response', grp_ref=0)
    print("OUTPUT", stats_fs_m0_resp_lm)
    print("OUTPUT", stats_fs_m0_resp_lm_hipamy)


    # Response * sex (Unscaled data, use tiv as regressor)
    stats_vbm_m0_resp_by_sex_lm = stats_lm(data_vbm_m0_, features_vbm_,
                formula="%s ~ response * sex + age + site + tiv",
                return_stats=["response", "age", "response:sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_vbm_m0_resp_by_sex_lm = multipletests_correct(stats_vbm_m0_resp_by_sex_lm, method="fdr_bh")
    stats_vbm_m0_resp_by_sex_lm["Modality"] = "VBM"
    stats_vbm_m0_resp_by_sex_lm = stats_vbm_m0_resp_by_sex_lm.sort_values(("p", "response:sex[T.M]"))
    print("OUTPUT", stats_vbm_m0_resp_by_sex_lm)

 
    stats_fs_m0_resp_by_sex_lm = stats_lm(data_fs_m0_, features_fs_,
                formula="%s ~ response * sex + age + site + tiv",
                return_stats=["response", "age", "response:sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_fs_m0_resp_by_sex_lm = multipletests_correct(stats_fs_m0_resp_by_sex_lm, method="fdr_bh")
    stats_fs_m0_resp_by_sex_lm["Modality"] = "FS"
    stats_fs_m0_resp_by_sex_lm = stats_fs_m0_resp_by_sex_lm.sort_values(("p", "response:sex[T.M]"))
    print("OUTPUT", stats_fs_m0_resp_by_sex_lm)


    # Scaled data
    target_tiv = 1500
    scaling_factor = target_tiv / data_vbm_m0_["tiv"]
    data_vbm_m0_scaled_ = data_vbm_m0_.copy()
    data_vbm_m0_scaled_[features_vbm_ + ["tiv"]] = data_vbm_m0_[features_vbm_+ ["tiv"]].mul(scaling_factor, axis=0)
    assert np.allclose(data_vbm_m0_scaled_.tiv, target_tiv)

    target_tiv = 1500000
    scaling_factor = target_tiv / data_fs_m0_["tiv"]
    data_fs_m0_scaled_ = data_fs_m0_.copy()
    data_fs_m0_scaled_[features_fs_] = data_fs_m0_[features_fs_].mul(scaling_factor, axis=0)
    assert np.allclose(data_fs_m0_scaled_.tiv, target_tiv)


    # Scaled data: same analysis without tiv as regressor
    stats_vbm_m0_scaled_resp_lm = stats_lm(data_vbm_m0_scaled_, features_vbm_,
                formula="%s ~ response + sex + age + site",
                return_stats=["response", "age", "sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_vbm_m0_scaled_resp_lm = multipletests_correct(stats_vbm_m0_scaled_resp_lm, method="fdr_bh")
    stats_vbm_m0_scaled_resp_lm["Modality"] = "VBM"
    stats_vbm_m0_scaled_resp_lm = stats_vbm_m0_scaled_resp_lm.sort_values(("p", "response"))
    print("OUTPUT", stats_vbm_m0_scaled_resp_lm)

    stats_fs_m0_scaled_resp_lm = stats_lm(data_fs_m0_scaled_, features_fs_,
                formula="%s ~ response + sex + age + site",
                return_stats=["response", "age", "sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_fs_m0_scaled_resp_lm = multipletests_correct(stats_fs_m0_scaled_resp_lm, method="fdr_bh")
    stats_fs_m0_scaled_resp_lm["Modality"] = "FS"
    stats_fs_m0_scaled_resp_lm = stats_fs_m0_scaled_resp_lm.sort_values(("p", "response"))
    print("OUTPUT", stats_fs_m0_scaled_resp_lm)


    # Response * sex interaction (scaled, no tiv)
    stats_vbm_m0_scaled_resp_by_sex_lm = stats_lm(data_vbm_m0_scaled_, features_vbm_,
                formula="%s ~ response * sex + age + site",
                return_stats=["response", "age", "response:sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_vbm_m0_scaled_resp_by_sex_lm = multipletests_correct(stats_vbm_m0_scaled_resp_by_sex_lm, method="fdr_bh")
    stats_vbm_m0_scaled_resp_by_sex_lm["Modality"] = "VBM"
    stats_vbm_m0_scaled_resp_by_sex_lm = stats_vbm_m0_scaled_resp_by_sex_lm.sort_values(("p", "response"))
    print("OUTPUT", stats_vbm_m0_scaled_resp_by_sex_lm)

    stats_fs_m0_scaled_resp_by_sex_lm = stats_lm(data_fs_m0_scaled_, features_fs_,
                formula="%s ~ response * sex + age + site",
                return_stats=["response", "age", "response:sex[T.M]"]).pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_fs_m0_scaled_resp_by_sex_lm = multipletests_correct(stats_fs_m0_scaled_resp_by_sex_lm, method="fdr_bh")
    stats_fs_m0_scaled_resp_by_sex_lm["Modality"] = "FS"
    stats_fs_m0_scaled_resp_by_sex_lm = stats_fs_m0_scaled_resp_by_sex_lm.sort_values(("p", "response"))
    print("OUTPUT", stats_fs_m0_scaled_resp_by_sex_lm)

    # Sex-stratified: stat_vbm_change_by_sex for ROI
    rois = FEATURES_HIPAMY_VBM_ #[lookup[r] for r in FEATURES_HIPAMY_VBM]    
    formula = "%s ~ response + sex + age + site + tiv"
    ivars = [ivar.strip() for ivar in formula.split("~")[1].split("+")]
    data_vbm_m0_adj = partial_residuals(data_vbm_m0_, rois, formula, res_vars=["sex[T.M]", 'response'])
    data_vbm_m0_adj = extract_adjusted(data_vbm_m0_adj, dvars=rois, ivars=ivars)
    

    stats_vbm_m0_by_sex_twosample = pd.concat(
        [twosample_stats(df_sex, cols=rois, grp='response').assign(sex=sex)
         for sex, df_sex in data_vbm_m0_adj.groupby("sex")],
        ignore_index=True,
    ).pivot(index="dvar", columns="sex")
    
    print('OUTPUT:', stats_vbm_m0_by_sex_twosample)


    # Brain map Glassview: VBM M0 Response (FDR < 0.05)
    # Import for brainmap
    import importlib, plots_sara
    importlib.reload(plots_sara)
    from plots_sara import plot_glassbrain
    import matplotlib.pyplot as plt

    # FDR
    sig = stats_vbm_m0_resp_lm[stats_vbm_m0_resp_lm[("p_corr", "response")] < 0.05]
    sig = sig[~sig.index.str.contains("White_Matter")]

    # Proportion of decrease to add in the title
    avg_hip = sig.loc[sig.index.str.contains("Hippocampus"), ("prop_change", "response")].mean() * 100
    avg_amy = sig.loc[sig.index.str.contains("Amygdala"), ("prop_change", "response")].mean() * 100
        
    val = sig[("t", "response")]
    dict_plot = dict(zip(sig.index.map(lookup_vbm_revert), val))
    threshold = val.abs().min() - val.abs().min() / 100
    title = (
        f"VBM M0: Response(T) (FDR < 0.05)\n"
        f"Responders: {avg_hip:.2f}% in hippocampus and {avg_amy:.2f}% in amygdala"
    )
    p = plot_glassbrain(dict_plot=dict_plot, title=title, threshold=threshold)
    print("OUTPUT:", "reports/statistics_m0_t_response_fdr.*")
    plt.gcf().savefig("reports/statistics_m0_t_response_fdr.png", dpi=150, bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_t_response_fdr.pdf", bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_t_response_fdr.svg", bbox_inches="tight")
    p.show()

    val = sig[("prop_change", "response")] * 100
    dict_plot = dict(zip(sig.index.map(lookup_vbm_revert), val))
    threshold = val.abs().min() - val.abs().min() / 100
    title = (
        f"VBM M0: Response (Coef % of mean) (FDR < 0.05)\n"
        f"Responders: {avg_hip:.2f}% in hippocampus and {avg_amy:.2f}% in amygdala"
    )
    p = plot_glassbrain(dict_plot=dict_plot, title=title, threshold=threshold)
    print("OUTPUT:", "reports/statistics_m0_coef_div_mean_response_fdr.*")
    plt.gcf().savefig("reports/statistics_m0_coef_div_mean_response_fdr.png", dpi=150, bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_coef_div_mean_response_fdr.pdf", bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_coef_div_mean_response_fdr.svg", bbox_inches="tight")
    p.show()
    
    # Uncorrected
    sig = stats_vbm_m0_resp_lm[stats_vbm_m0_resp_lm[("p", "response")] < 0.05]
    sig = sig[~sig.index.str.contains("White_Matter")]
    n_sig = len(sig)
    # Proportion of decrease to add in the title
    prop_reduction_in_reponser = np.sum(sig[('coef', 'response')] < 0) / len(sig[('coef', 'response')])
    change_mean = sig[("prop_change", "response")].mean() * 100
    change_std = sig[("prop_change", "response")].std() * 100
    
    val = sig[("t", "response")]
    dict_plot = dict(zip(sig.index.map(lookup_vbm_revert), val))
    threshold = val.abs().min()- val.abs().min() / 100
    title = (
        f"VBM M0: Response(T) (P-value < 0.05, uncorrected)\n"
        f"{n_sig} ROIs ({prop_reduction_in_reponser:.0%} are negatives) "
        f"Responsers have {change_mean:.2f}%±{change_std:.2f}% (mean±std)"
    )
    p = plot_glassbrain(dict_plot=dict_plot, title=title, threshold=threshold)
    print("OUTPUT:", "reports/statistics_m0_t_response_uncorr.*")
    plt.gcf().savefig("reports/statistics_m0_t_response_uncorr.png", dpi=150, bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_t_response_uncorr.pdf", bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_t_response_uncorr.svg", bbox_inches="tight")
    p.show()

    val = sig[("prop_change", "response")] * 100
    dict_plot = dict(zip(sig.index.map(lookup_vbm_revert), val))
    threshold = val.abs().min()- val.abs().min() / 100
    title = (
        f"VBM M0: Response (Coef % of mean) (P-value < 0.05, uncorrected)\n"
        f"{prop_reduction_in_reponser:.0%} are < 0. "
        f"Responsers have {change_mean:.2f}%±{change_std:.2f}% (mean±std)"
    )
    p = plot_glassbrain(dict_plot=dict_plot, title=title, threshold=threshold)
    print("OUTPUT:", "reports/statistics_m0_coef_div_mean_response_uncorr.*")
    plt.gcf().savefig("reports/statistics_m0_coef_div_mean_response_uncorr.png", dpi=150, bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_coef_div_mean_response_uncorr.pdf", bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_m0_coef_div_mean_response_uncorr.svg", bbox_inches="tight")
    p.show()
    
    
    # Average LR and Plots of ROI volumes adjusted for nuisance covariates 
    rois = ["Hippocampus", "Amygdala"]
    formula = "%s ~ response + sex + age + site + tiv"
    ivars = [ivar.strip() for ivar in formula.split("~")[1].split("+")]

    data_vbm_m0_lr = average_lr_rois(data_vbm_m0, rois)[rois + ["tiv"] + ["response", "sex", "age", "site"]]
    data_vbm_m0_lr = partial_residuals(data_vbm_m0_lr, rois, formula, res_vars=["response", "sex"])
    data_vbm_m0_lr = extract_adjusted(data_vbm_m0_lr, rois, ivars=["response", "sex"])

    stats_vbm_m0_lr_twosample = twosample_stats(data_vbm_m0_lr, cols=rois, grp='response')
    print('OUTPUT:', stats_vbm_m0_lr_twosample)

    stats_vbm_m0_lr_by_sex_twosample = pd.concat(
        [twosample_stats(df_sex, cols=rois, grp='response').assign(sex=sex)
         for sex, df_sex in data_vbm_m0_lr.groupby("sex")],
        ignore_index=True,
    ).pivot(index="dvar", columns="sex")
    
    print('OUTPUT:', stats_vbm_m0_lr_by_sex_twosample)

    fig, axes = violinplot_adjusted(data_vbm_m0_lr, rois, hue="response",
                                    split=True, swarm=True)
    fig.savefig("reports/statistics_m0_vbm_lr_adjusted_response.svg", bbox_inches="tight")

    fig, axes = violinplot_adjusted(data_vbm_m0_lr, rois, hue="response",
                                    split=True, swarm=True,
                                    hue_statannotations=True,
                                    annot_text_format="star+stat")
    fig.savefig("reports/statistics_m0_vbm_lr_adjusted_response_annotated.svg", bbox_inches="tight")

    fig, axes = violinplot_adjusted(data_vbm_m0_lr, rois,
        x="sex", hue="response", swarm=True, split=True,
        hue_statannotations=True, annot_text_format="star+stat")
    fig.savefig("reports/statistics_m0_vbm_lr_adjusted_sex_by_response.svg", bbox_inches="tight")


    excel_path = "reports/statistics_m0_response.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet, df in [
            ("vbm_m0_resp_tests",             stats_vbm_m0_resp_tests),
            ("vbm_m0_resp_lm",                stats_vbm_m0_resp_lm),
            ("fs_m0_resp_lm",                 stats_fs_m0_resp_lm),
            ("fs_m0_resp_lm_hipamy",          stats_fs_m0_resp_lm_hipamy),         
            ("vbm_m0_resp_by_sex_lm",         stats_vbm_m0_resp_by_sex_lm),
            ("vbm_m0_lr_twosample",           stats_vbm_m0_lr_twosample),
            ("vbm_m0_lr_by_sex_twosample",    stats_vbm_m0_lr_by_sex_twosample),
            ("fs_m0_resp_by_sex_lm",          stats_fs_m0_resp_by_sex_lm),
            ("vbm_m0_by_sex_twosample",       stats_vbm_m0_by_sex_twosample),            
            ("vbm_m0_scaled_resp_lm",         stats_vbm_m0_scaled_resp_lm),
            ("fs_m0_scaled_resp_lm",          stats_fs_m0_scaled_resp_lm),
            ("vbm_m0_scaled_resp_by_sex_lm",  stats_vbm_m0_scaled_resp_by_sex_lm),
            ("fs_m0_scaled_resp_by_sex_lm",   stats_fs_m0_scaled_resp_by_sex_lm),
        ]:
            out = df.copy()
            out.columns = [f"{c[1]}_{c[0]}" if isinstance(c, tuple) else c for c in out.columns]
            out.to_excel(writer, sheet_name=sheet, index=True)
    print(f"Saved {excel_path}")
    """
        PROMPT TO GENERATE THE TABLES
    
    M0 ~ Response
    Read the attached Excel file: statistics_m0_response.xlsx. Read the sheets stats_vbm_m0_resp_lm for VBM and fs_m0_resp_lm_hipamy for Freesurfer.
    Create a table with columns:
    1. Modality in VBM or FreeSurfer from column _Modality
    2. ROI from column dvar, ex: Right_Hippocampus_GM_Vol => Hippocampus
    3. Side from column dvar  ex: Right_Hippocampus_GM_Vol => Right
    4. Response with two subcolumns:
       - Coef (% of mean), T from columns response_prop_change*100, response_t (significance annotation from FDR corrected p-value: response_p_corr)
    5. Sex[Male] with two subcolumns:
       - Coef (% of mean), T from columns sex[T.M]_prop_change*100, sex[T.M]_t (significance annotation from FDR corrected p-value: sex[T.M]_p_corr)
    6. Age (T) with two subcolumns:
       - Coef (% of mean), T from columns age_prop_change*100, age_t (significance annotation from FDR corrected p-value: age_p_corr)
    
    Start with all VBM rows, then the FreeSurfer ones
    Within each modality order row by Hippocampus Left / Right then Amygdala Left/Right
    For VBM keep only significant features for response: response_p_corr < 0.05
    Use only 2 decimals
    
    Propose the Table title and caption knowing that:
    - the linear model was : Feature ~ response + sex + age + site + tiv
    - Explained what is Coef (% of mean) is β / ȳ (computed in reference non responder group) * 100, ie (%) "Relative effect size" such that it can be interpreted as "responders show an X% lower volume relative to non-responders."
    The significance is FDR corrected
    For the FreeSurfer feature we tested only the four volume from bilateral Amygdala, Hippocampus
    
    Change ~ Response sex-stratified
    Read the attached Excel file:statistics_m0_response.xlsx. Read the vbm_m0_by_sex_twosample sheet.
    Create a markdown table with columns with Feature, Cohens'd (Subdivided M/F), T Subdivided M/F), P-value Subdivided M/F)
    Put in bold if significant.
    Create a table with columns:
    - Feature from dvar column
    - Cohens'd (Subdivided M/F) from F_cohens_d	M_cohens_d
    - T Subdivided M/F) from  from F_welch_t	M_welch_t
    - P-value Subdivided M/F) from F_welch_p	M_welch_p
    Exprime proportion in %. Use only 2 decimals for T-value and 4 four P-values
    """
# %%
def stats_vbm_change_lm():
    # Import for brainmap
    import importlib, plots_sara
    importlib.reload(plots_sara)
    from plots_sara import plot_glassbrain
    import matplotlib.pyplot as plt
    
    # Load data
    data_vbm_m0, data_vbm_change, data_fs, data_vbm_fs = load_data()
    data_vbm_change["response"] = data_vbm_change["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})

    # keeep only GM features
    features_vbm_gm = [c for c in data_vbm_change.columns if c not in ["age", "sex", "response", "site"] and "GM" in c]

    # --------------------------------------------------------------------------
    # 1. stats_vbm_change_lm = M3-M0 ~ 1 + C(age) + C(sex) + C(site)
    
    # Design matrix of centered variables C(age) + C(sex) + C(site)
    dummies = pd.get_dummies(data_vbm_change[["sex", "site", "response"]], drop_first=True, dtype=float)
    design = pd.concat([data_vbm_change[["age"]], dummies], axis=1)
    design -= design.mean()
    data_vbm_change_ = pd.concat([data_vbm_change[features_vbm_gm], design], axis=1)
    print(data_vbm_change_[design.columns].mean())
    lookup, lookup_revert = make_varname_lookup(data_vbm_change_.columns)
    data_vbm_change_ = data_vbm_change_.rename(columns=lookup)
    design = design.rename(columns=lookup)

    # 1.1 Fit models M3-M0 ~ 1 + C(age) + C(sex) + C(site)    
    formula = "%s ~ " + "+".join([c for c in design.columns if  c != "response"])
    dvars = [lookup[f] for f in features_vbm_gm]
    # '%s ~ sex + age + site'
    stats_vbm_change_lm = stats_lm(data_vbm_change_, dvars,
        formula=formula,
        return_stats=["Intercept", "age", "sex_M"])
    stats_vbm_change_lm["dvar"] = stats_vbm_change_lm["dvar"].map(lookup_revert)
    stats_vbm_change_lm = stats_vbm_change_lm.pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])

    # FDR
    from statsmodels.stats.multitest import multipletests
    p_cols = [c for c in stats_vbm_change_lm.columns if c[0] == "p"]
    for col in p_cols:
        _, p_corr, _, _ = multipletests(stats_vbm_change_lm[col], method="fdr_bh")
        stats_vbm_change_lm[("p_corr", col[1])] = p_corr

    # Add proportion of change = (M3 - M0) / M0    
    common_idx = data_vbm_change.index.intersection(data_vbm_m0.index)
    assert len(common_idx) == 91
    common_columns_ = data_vbm_change.columns.intersection(data_vbm_m0.columns)
    common_columns_ = [c for c in common_columns_ if c not in ['response', 'age', 'sex', 'site'] and "GM" in c]
    assert len(common_columns_) == 134
    assert set(common_columns_) == set(features_vbm_gm), diff_lists(common_columns_, features_vbm_gm)

    prop_change_vbm = data_vbm_change.loc[common_idx, features_vbm_gm] / data_vbm_m0.loc[common_idx, features_vbm_gm]
    prop_change_vbm = prop_change_vbm.mean()
    print(prop_change_vbm.describe())
    print(prop_change_vbm[FEATURES_HIPAMY_VBM])
    # Make sur prop is aligned to stats_vbm_change_lm
    prop_change_vbm = prop_change_vbm[stats_vbm_change_lm.index]
    stats_vbm_change_lm[("prop_change", "mean")] = prop_change_vbm

    stats_vbm_change_lm = stats_vbm_change_lm.sort_values(('p_corr', 'Intercept'))
    print('OUTPUT:', stats_vbm_change_lm)

    # 1.2 Brain map Glassview
    sig = stats_vbm_change_lm[stats_vbm_change_lm[('p_corr', 'Intercept')] < 0.05]
    val = sig[('t', 'Intercept')]
    # Plot t-map of intercept 
    dict_plot = dict(zip(sig.index, val))
    threshold = val.abs().min()
    p = plot_glassbrain(dict_plot=dict_plot, title="VBM M3-M0: Intercept (FDR < 0.05)", threshold=threshold)    
    print('OUTPUT:', "reports/statistics_change_intercept.*")
    plt.gcf().savefig("reports/statistics_change_intercept.png", dpi=150, bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_change_intercept.pdf", bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_change_intercept.svg", bbox_inches="tight")
    p.show()
    
    # Plot proportion of change
    pc = sig[("prop_change", "mean")]
    print(pc.describe())
    pc_mean = pc.mean()
    pc_max_roi = pc.abs().idxmax()
    pc_max_val = pc[pc_max_roi]
    title = (f"VBM (M3-M0)/M0 (FDR<0.05)\n"
             f"mean={pc_mean:.3f}  max={pc_max_val:.3f} ({pc_max_roi})")
    dict_plot = dict(zip(sig.index, pc))
    threshold = pc.min()
    p = plot_glassbrain(dict_plot=dict_plot, title=title, threshold=threshold)
    print('OUTPUT:', "reports/statistics_change_proportion.*")
    plt.gcf().savefig("reports/statistics_change_proportion.png", dpi=150, bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_change_proportion.pdf", bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_change_proportion.svg", bbox_inches="tight")
    p.show()

    # 1.4 sex-stratified changes: stat_vbm_change_by_sex for ROI
    # ROI = M3-M0 ~ 1 + C(age) + C(sex) + C(site)
    rois = [lookup[r] for r in FEATURES_HIPAMY_VBM]    
    # formula = '%s ~ age+sex_M+site_site_02+site_site_03+site_site_04+site_site_05+site_site_06+site_site_07+site_site_08+site_site_09+site_site_10+site_site_11+site_site_12+site_site_13+site_site_14+site_site_15+site_site_16'
    ivars = [ivar.strip() for ivar in formula.split("~")[1].split("+")]
    data_vbm_change_adj = partial_residuals(data_vbm_change_, rois, formula, res_vars=["sex_M"])
    data_vbm_change_adj = extract_adjusted(data_vbm_change_adj, dvars=rois, ivars=ivars)
    
    # sex column sex_M is [0.55, -0.45, ...] 1/0 centerred, convert to [M, F, ...] 
    sex_num = np.sort(data_vbm_change_adj.sex_M.unique())
    data_vbm_change_adj['sex'] = data_vbm_change_adj.sex_M.map({sex_num[0]:'F', sex_num[1]:'M'})

    stats_vbm_change_by_sex_onesample = pd.concat(
        [onesample_stats(df_sex, cols=rois).assign(sex=sex)
         for sex, df_sex in data_vbm_change_adj.groupby("sex")],
        ignore_index=True,
    ).pivot(index="dvar", columns="sex")
    
    print('OUTPUT:', stats_vbm_change_by_sex_onesample)

    # --------------------------------------------------------------------------
    # 2. M3-M0 ~ response
    dvars = [lookup[f] for f in features_vbm_gm]
    lookup, lookup_revert = make_varname_lookup(data_vbm_change.columns)
    data_vbm_change_ = data_vbm_change.rename(columns=lookup)

    # 2.1 Fit M3-M0 ~ 1 + response + age + sex + site
    formula = "%s ~ response + sex + age + site"# % 'Left_Hippocampus'
    stats_vbm_change_resp_lm = stats_lm(data_vbm_change_, dvars,
        formula=formula,
        return_stats=['response', 'sex[T.M]', 'age'])
    stats_vbm_change_resp_lm["dvar"] = stats_vbm_change_resp_lm["dvar"].map(lookup_revert)
    stats_vbm_change_resp_lm = stats_vbm_change_resp_lm.pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_vbm_change_resp_lm = stats_vbm_change_resp_lm.sort_values(('p', 'response'))

    # FDR
    from statsmodels.stats.multitest import multipletests
    p_cols = [c for c in stats_vbm_change_resp_lm.columns if c[0] == "p"]
    for col in p_cols:
        _, p_corr, _, _ = multipletests(stats_vbm_change_resp_lm[col], method="fdr_bh")
        stats_vbm_change_resp_lm[("p_corr", col[1])] = p_corr

    stats_vbm_change_resp_lm = stats_vbm_change_resp_lm.sort_values(('p_corr', 'response'))
    print('OUTPUT:', stats_vbm_change_resp_lm)
    
    # 2.2 Brain map Glassview    
    sig = stats_vbm_change_resp_lm[stats_vbm_change_resp_lm[('p', 'response')] < 0.05]
    dict_plot = dict(zip(sig.index, sig[('t', 'response')]))
    print('OUTPUT:', "reports/statistics_change_response.*")
    p = plot_glassbrain(dict_plot=dict_plot, title="VBM M3-M0: Response (P-value < 0.05, uncorrected)")
    plt.gcf().savefig("reports/statistics_change_response.png", dpi=150, bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_change_response.pdf", bbox_inches="tight")
    plt.gcf().savefig("reports/statistics_change_response.svg", bbox_inches="tight")
    p.show()
    
    # 2.3 sex-response interaction in changes: stat_vbm_change_by_sex for ROI
    formula = "%s ~ response * sex + age + site"# % 'Left_Hippocampus'
    stats_vbm_change_resp_sex_lm = stats_lm(data_vbm_change_, dvars,
        formula=formula,
        return_stats=['response:sex[T.M]', 'response', 'sex[T.M]', 'age'])
    stats_vbm_change_resp_sex_lm["dvar"] = stats_vbm_change_resp_sex_lm["dvar"].map(lookup_revert)
    stats_vbm_change_resp_sex_lm = stats_vbm_change_resp_sex_lm.pivot(index="dvar", columns="ivar", values=["coef", "t", "p"])
    stats_vbm_change_resp_sex_lm = stats_vbm_change_resp_sex_lm.sort_values(('p', 'response:sex[T.M]'))
    
    print('OUTPUT:', stats_vbm_change_resp_sex_lm)

    
    # 2.4 sex-stratified changes: stats_vbm_change_resp_by_sex_onesample for ROI
    # ROI = M3-M0 ~ 1 + C(age) + C(sex) + C(site)
    formula = "%s ~ response + sex + age + site"# % 'Left_Hippocampus'
    rois = [lookup[r] for r in FEATURES_HIPAMY_VBM]    
    # formula = '%s ~ age+sex_M+site_site_02+site_site_03+site_site_04+site_site_05+site_site_06+site_site_07+site_site_08+site_site_09+site_site_10+site_site_11+site_site_12+site_site_13+site_site_14+site_site_15+site_site_16'
    ivars = [ivar.strip() for ivar in formula.split("~")[1].split("+")]
    data_vbm_change_resp_adj = partial_residuals(data_vbm_change_, rois, formula, res_vars=["response", "sex[T.M]"])
    data_vbm_change_resp_adj = extract_adjusted(data_vbm_change_resp_adj, dvars=rois, ivars=ivars)
    
    # sex column sex_M is [0.55, -0.45, ...] 1/0 centerred, convert to [M, F, ...] 
    #sex_num = np.sort(data_vbm_change_adj.sex_M.unique())
    #data_vbm_change_adj['sex'] = data_vbm_change_adj.sex_M.map({sex_num[0]:'F', sex_num[1]:'M'})

    stats_vbm_change_resp_by_sex_onesample = pd.concat(
        [twosample_stats(df_sex, cols=rois, grp='response').assign(sex=sex)
         for sex, df_sex in data_vbm_change_resp_adj.groupby("sex")],
        ignore_index=True,
    ).pivot(index="dvar", columns="sex")
    
    print('OUTPUT:', stats_vbm_change_resp_by_sex_onesample)
    

    # Save it 
    excel_path = "reports/statistics_change.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet, df in [
            ("vbm_change_lm", stats_vbm_change_lm),
            ("vbm_change_by_sex_onesample", stats_vbm_change_by_sex_onesample),
            ("vbm_change_resp_lm", stats_vbm_change_resp_lm),
            ("stats_vbm_change_resp_sex_lm", stats_vbm_change_resp_sex_lm),
            ("vbm_change_resp_by_sex", stats_vbm_change_resp_by_sex_onesample),
        ]:
            out = df.copy()
            out.columns = [f"{c[1]}_{c[0]}" if isinstance(c, tuple) else c for c in out.columns]
            out.to_excel(writer, sheet_name=sheet, index=True)
    print(f"Saved {excel_path}")

    """
    PROMPT TO GENERATE THE TABLES
    
    Change ~ Interpect (Global increase)
    Read the attached Excel file:statistics_change.xlsx. Read the vbm_change_lm sheet.
    Use attached "lobes_Neuromorphometrics_with_dfROI_correspondencies.csv" file to remap Feature name from [Side] + column "ROI_Neuromorphometrics_labels" to column "ROIname"
    Create a table with columns with Feature names and T-values. T-values are annotated with signiticance notation *<0.05, etc.
    Put in bold if significant.
    Create a table with columns:
    - Feature => rename feature using the mapping provided in "lobes_Neuromorphometrics_with_dfROI_correspondencies.csv" file. "lobes_Neuromorphometrics_with_dfROI_correspondencies.csv" file to remap Feature name from [Side] + column "ROI_Neuromorphometrics_labels" to column "ROIname"
    - Intercept (T) from Intercept_t, use significance annotation from Intercept_p_corr
    - Proportion of Change from mean_prop_change
    - Age (T) from age_t, use significance annotation from age_p_corr
    - Sex[Male] (T)  from sex_M_t, use significance annotation from sex_M_p_corr
    Exprime proportion in %. Use only 2 decimals
    
    Change ~ Response sex-stratified
    Read the attached Excel file:statistics_change.xlsx. Read the vbm_change_resp_by_sex sheet.
    Create a markdown table with columns with Feature, Cohens'd (Subdivided M/F), T Subdivided M/F), P-value Subdivided M/F)
    Put in bold if significant.
    Create a table with columns:
    - Feature from dvar column
    - Cohens'd (Subdivided M/F) from F_cohens_d	M_cohens_d
    - T Subdivided M/F) from  from F_welch_t	M_welch_t
    - P-value Subdivided M/F) from F_welch_p	M_welch_p
    Exprime proportion in %. Use only 2 decimals for T-value and 4 four P-values
    """

# # %%
# def stat_vbm_change_response():
#     import matplotlib.pyplot as plt
#     plt.style.use('seaborn-v0_8-whitegrid')
    
#     # Load data
#     data_vbm_m0, data_vbm_change, data_fs, data_vbm_fs = load_data()
#     data_vbm_change["response"] = data_vbm_change["response"].map({'NR': 0, 'PaR': 0, 'GR': 1})
#     demo_cols = ["age", "sex", "response", "site"]
#     data_vbm_change = data_vbm_change[demo_cols + FEATURES_HIPAMY_VBM]
    
#     lookup, lookup_revert = make_varname_lookup(data_vbm_change.columns, rmsuffix='_GM_Vol')
#     data_vbm_change = data_vbm_change.rename(columns=lookup)
    
#     # LM
#     formula = "%s ~ response + sex + age + site"# % 'Left_Hippocampus'
#     return_stats = ['response', 'sex[T.M]', 'age']
#     lm_stats_ = stats_lm(data=data_vbm_change, features=rois, formula=formula, return_stats=return_stats).set_index("dvar")
#     print(lm_stats_)

#     # LM with Interaction
#     formula = "%s ~ response * sex + age + site"# % 'Left_Hippocampus'
#     return_stats = ['response', 'response:sex[T.M]']
#     lm_interaction_stats_ = stats_lm(data=data_vbm_change, features=rois, formula=formula, return_stats=return_stats).set_index("dvar")
#     print(lm_interaction_stats_)
    
    
#     # Plots of ROI volumes adjusted for nuisance covariates
#     rois = [lookup[r] for r in FEATURES_HIPAMY_VBM]    
#     formula = "%s ~ response + sex + age + site"
#     ivars = [ivar.strip() for ivar in formula.split("~")[1].split("+")]
#     data_vbm_change_adj = partial_residuals(data_vbm_change, rois, formula, res_vars=["response", "sex"])
#     data_vbm_change_plot = extract_adjusted(data_vbm_change_adj, rois, ivars=["response", "sex"])
    
#     # per ROI
#     fig, axes = violinplot_adjusted(data_vbm_change_plot, rois,
#                                     split=False, swarm=True, sharey=True, hline=0,
#                                     annot_text_format="star+stat")
#     fig.savefig("reports/statistics_vbm_change.svg", bbox_inches="tight")
#     plt.show()
#     adj_vbm_change_onesample_stats = onesample_stats(data_vbm_change_plot, cols=rois)
#     print(adj_vbm_change_onesample_stats)

#     # Per ROI per response
#     fig, axes = violinplot_adjusted(data_vbm_change_plot, rois, hue="response",
#                                     split=True, swarm=True, hline=0,
#                                     #hue_statannotations=True,
#                                     sharey=True, annot_text_format="star+stat")
#     fig.savefig("reports/statistics_vbm_change_by_response.svg", bbox_inches="tight")
#     plt.show()
#     adj_vbm_change_by_response_twosample_stats = twosample_stats(data=data_vbm_change_plot, grp="response", cols=rois)
#     print(adj_vbm_change_by_response_twosample_stats)

#     # Per ROI sex per per response
#     fig, axes = violinplot_adjusted(data_vbm_change_plot, rois, x="sex", hue="response",
#                                     split=True, swarm=True, hline=0,
#                                     hue_statannotations=True, sharey=True, annot_text_format="star+stat")
#     fig.savefig("reports/statistics_vbm_change_by_sex_response.svg", bbox_inches="tight")
#     plt.show()
#     adj_vbm_change_by_sex_response_twosample_stats = pd.concat(
#         [twosample_stats(df_sex, grp="response", cols=rois).assign(sex=sex)
#          for sex, df_sex in data_vbm_change_plot.groupby("sex")],
#         ignore_index=True,
#     )
#     print(adj_vbm_change_by_sex_response_twosample_stats)

#     excel_path = "reports/statistics_vbm_change_response_hipamy.xlsx"
#     with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
#         for sheet, df in [
#             ("lm_stats", lm_stats_),
#             ("lm_interaction_stats", lm_interaction_stats_),
#             ("onesample_stats", adj_vbm_change_onesample_stats),
#             ("twosample_by_response", adj_vbm_change_by_response_twosample_stats),
#             ("twosample_by_sex_response", adj_vbm_change_by_sex_response_twosample_stats),
#         ]:
#             df.to_excel(writer, sheet_name=sheet, index=True)
#     print(f"Saved {excel_path}")

#     """ PROMPT
#     Read the attached Excel file:statistics_vbm_change_response_hipamy.xlsx. Read the lm_stats sheet.
#     Create a markdown table with columns with Feature names and T-values. T-values are annotated with signiticance notation *<0.05, etc.
#     Put in bold if significant.
#     Create a table, onbe row per var with columns:
#     - Feature => var column
#     - Age (T) with significance annotation from t and p where ivar ==  age
#     - Sex[Male] with significance annotation from t and p where ivar == sex[T.M]
#     - Response (T) with significance annotation from t and p where ivar == response
    
#     Use only 2 decimals
#     """

# %% ===========================================================================
if __name__ == '__main__':
    stats_m0()
    # stats_vbm_change_lm()