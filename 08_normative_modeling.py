import pandas as pd
import numpy as np
import sys
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import SplineTransformer
from sklearn.compose import ColumnTransformer
from scipy.optimize import minimize
from scipy.stats import norm


# for residualization on site
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

# inputs
FOUR_REGIONS_OF_INTEREST = ["lHip_GM_Vol","rHip_GM_Vol","lAmy_GM_Vol","rAmy_GM_Vol"]
FOUR_REGIONS_OF_INTEREST_LONG = ['Left Hippocampus_GM_Vol','Right Hippocampus_GM_Vol','Left Amygdala_GM_Vol', 'Right Amygdala_GM_Vol']
ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
PATH_TO_DATA_OPENBHB = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/"
OPENBHB_DATAFRAME = DATA_DIR+"OpenBHB_roi.csv"
RLINK_DATAFRAME_M00_M03 = DATA_DIR + "df_ROI_age_sex_site_M00_M03_v4labels.csv"
RLINK_DATAFRAME_ALL_M00 = DATA_DIR + "df_ROI_age_sex_site_M00_v4labels.csv"



def warp_sinh_arcsinh(y, epsilon=0.0, b=1.0):
    # Warping function g(y) = sinh(b * arcsinh(y) + epsilon * b)
    return np.sinh(b * np.arcsinh(y) + epsilon * b)

def inv_warp_sinh_arcsinh(gy, epsilon=0.0, b=1.0):
    # Inverse warping g^{-1}(y) = sinh((arcsinh(gy) - epsilon * b)/b)
    return np.sinh((np.arcsinh(gy) - epsilon * b) / b)

def log_likelihood_sinh_arcsinh(params, y):
    epsilon, b = params
    if b <= 0:
        return np.inf  # log-likelihood undefined for non-positive beta

    # Warped data
    g_y = warp_sinh_arcsinh(y, epsilon, b)

    # Assume standard normal after warping
    logpdf = norm.logpdf(g_y)

    # Compute derivative dg/dy = b * cosh(b * arcsinh(y) + epsilon * b) / sqrt(1 + y^2)
    dg_dy = b * np.cosh(b * np.arcsinh(y) + epsilon * b) / np.sqrt(1 + y**2)

    # Change-of-variables log-likelihood
    return -np.sum(logpdf + np.log(dg_dy))  # negative for minimization

def get_M0_M3_df_for_chosen_label(label_value, label_type="int", comparable_dataframes_N91=True):
    """
    label_type : either 'int' or 'str' (int refers to classification labels 0 or 1, 
                str to string labels GR, PaR, or NR)

    returns two dataframes of ROIs at baseline (M0) and 3 months after li intake (M3) 
    
    if comparable_dataframes_N91 is True, the rows of the two dataframes
         correspond to the same participants (same order)
        that way, they are comparable row wise (N=91)
    if comparable_dataframes_N91 is False, the rows aren't comparable and the two dataframes are not the same length,
        but we get more participants in the dataframe at baseline (N=117)
    """

    assert label_type in ["str", "int"],"wrong label type"
    if label_type=="str": assert label_value in ["GR", "PaR", "NR"],"wrong label"
    if label_type =="int":assert label_value ==1 or label_value ==0," wrong label "

    df = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    if not comparable_dataframes_N91 : 
        df_M0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    if label_type=="int": 
        df["y"] = df["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
        if not comparable_dataframes_N91 : df_M0["y"] = df_M0["y"].replace({"GR": 1, "PaR": 0, "NR": 0})

    # get participant ids matching the given label (0=GR, 1=PaR/NR)
    participant_ids = df.loc[df["y"] == label_value, "participant_id"]

    # filder df by session to get m0 and m3 dataframes
    def filter_session(session_name):
        df_sess = df[df["session"] == session_name]
        df_sess = df_sess[df_sess["participant_id"].isin(participant_ids)].reset_index(drop=True)
        return df_sess

    df_M3 = filter_session("M03")
    if comparable_dataframes_N91: df_M0 = filter_session("M00")
    else : 
        df_M0=df_M0[df_M0["y"]==label_value]
        df_M0["session"]=["M00"]*len(df_M0)

    return df_M0, df_M3

def plot_spline_knots_effect(df_train, roi, sex_value=1, knots_list=[3,5,7,9]):
    """
    Plot spline fits of ROI vs age for different number of knots.
    
    Parameters:
    - df_train : training dataframe with 'age', 'sex', and ROI columns
    - roi : string, name of ROI column to plot
    - sex_value : int (0 or 1), sex to fix for prediction (default=1)
    - knots_list : list of int, number of knots to try
    """

    X_age = df_train[['age']].values
    y = df_train[roi].values

    age_grid = np.linspace(df_train['age'].min(), df_train['age'].max(), 200).reshape(-1,1)

    plt.figure(figsize=(10,6))
    plt.scatter(df_train['age'], y, alpha=0.4, label='Data')

    for n_knots in knots_list:
        # Build spline transformer + model pipeline
        spline = SplineTransformer(degree=3, n_knots=n_knots)
        X_spline = spline.fit_transform(X_age)
        # Add sex column fixed to sex_value
        X_design = np.hstack([X_spline, np.full((X_spline.shape[0],1), sex_value)])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_design)

        model = BayesianRidge()
        model.fit(X_scaled, y)

        # Predict on grid
        age_grid_spline = spline.transform(age_grid)
        X_grid = np.hstack([age_grid_spline, np.full((age_grid_spline.shape[0],1), sex_value)])
        X_grid_scaled = scaler.transform(X_grid)
        y_pred = model.predict(X_grid_scaled)

        plt.plot(age_grid, y_pred, label=f'n_knots={n_knots}')

    plt.xlabel('Age')
    plt.ylabel(roi)
    plt.title(f'Spline fit of {roi} vs Age for sex={sex_value}')
    plt.legend()
    plt.show()


def strat_stats(subset, name):
    mean_age = subset['age'].mean()
    prop_female = subset['sex'].mean()  # sex==1 is female
    print(f"{name} set - mean age: {mean_age:.2f}, proportion female (sex==1): {prop_female:.2f}")


def stratified_split(df, test_size=0.2, random_state=42, verbose=False):
    # Bin continuous 'age' into quantiles for stratification
    df = df.copy()
    df['age_bin'] = pd.qcut(df['age'], q=3, duplicates='drop')

    # Bin site: group rare sites
    site_counts = df['site'].value_counts()
    min_site_size=70
    rare_sites = site_counts[site_counts < min_site_size].index
    df['site_binned'] = df['site'].replace(rare_sites, 'other')

    df['strata'] = (
        df['age_bin'].astype(str) + "_" +
        df['sex'].astype(str) + "_" +
        df['site_binned'].astype(str)
    )

    # Perform stratified split
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df['strata'],
        random_state=random_state
    )

    df_train = df.loc[train_idx].drop(columns=['age_bin', 'strata'])
    df_test = df.loc[test_idx].drop(columns=['age_bin', 'strata'])

    if verbose:
        strat_stats(df_train, "Train")
        strat_stats(df_test, "Test")

    return df_train, df_test


def train_normative_model_with_openBHB(roi, bsplines=True, warp=False, residualize=False):
    df_openbhb = pd.read_csv(OPENBHB_DATAFRAME)
    # print("dataframe OpenBHB rois ...\n",df_openbhb)

    if residualize: 
        print("Residualizing on site")
        residualizer = Residualizer(
            data=df_openbhb,
            formula_res="site",
            formula_full= "site + age + sex"
        )
        Zres = residualizer.get_design_mat(df_openbhb)
        # select roi features
        roi_openbhb = [r for r in list(df_openbhb.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
        X_values = df_openbhb[roi_openbhb].values
        # fit and apply residualization
        residualizer.fit(X_values, Zres)
        X_values = residualizer.transform(X_values, Zres)
        df_openbhb_new = pd.DataFrame(X_values, columns=roi_openbhb, index=df_openbhb.index)
        df_openbhb_new[["age","sex","site"]] = df_openbhb[["age","sex","site"]]
        df_openbhb_new.insert(0, 'participant_id', df_openbhb["participant_id"])   
        df_train, df_test = stratified_split(df_openbhb_new)   

    else : df_train, df_test = stratified_split(df_openbhb)


    # plot_spline_knots_effect(df_train, roi, sex_value=0)
    # plot_spline_knots_effect(df_train, roi, sex_value=1)

    # Create spline basis for age with 3 degrees of freedom (knots)
    # This creates new spline basis columns for age
    X_tr = df_train[['age', 'sex']]
    X_te = df_test[['age', 'sex']]
    y_tr =  df_train[roi].values
    y_te =  df_test[roi].values

    # Warp target
    epsilon_opt = 0.0
    beta_opt = 1.0

    if warp:
        initial_params = [epsilon_opt, beta_opt]
        res = minimize(log_likelihood_sinh_arcsinh, initial_params, args=(y_tr,), method='L-BFGS-B', bounds=[(-2, 2), (1e-3, 10)])
        epsilon_opt, beta_opt = res.x
        # print(f"Optimal epsilon: {epsilon_opt}, beta: {beta_opt}")
        y_tr = warp_sinh_arcsinh(y_tr, epsilon=epsilon_opt, b=beta_opt)


    if bsplines:
        preprocessor = ColumnTransformer([
            ('spline_age', SplineTransformer(degree=3, n_knots=3), ['age']),
            ('passthrough_sex', 'passthrough', ['sex'])
        ])

    # fit model
    pipeline = make_pipeline(
        preprocessor,
        StandardScaler(),
        BayesianRidge()
    )

    # model = BayesianRidge() # works the same as ridge with gridsearch
    
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    # Inverse warp to get predictions in original space
    if warp : y_pred = inv_warp_sinh_arcsinh(y_pred, epsilon=epsilon_opt, b=beta_opt)

    r2 = r2_score(y_te, y_pred)
    print(f"Test R² score: {r2:.4f}")

    # compute residuals and standard deviation of residuals
    residuals = y_te - y_pred
    resid_std = residuals.std(ddof=1)  # use ddof=1 for sample std

    z_scores_test = (y_te - y_pred) / resid_std
    # Basic diagnostics on z-scores
    # mean and std should be around 0 and 1 respectively
    print(f"Z-scores on test set: mean={np.mean(z_scores_test):.3f}, std={np.std(z_scores_test, ddof=1):.3f}")

    # Plot histogram to check distribution of z-scores
    # plt.hist(z_scores_test, bins=30, alpha=0.7)
    # plt.title(f"Z-score Distribution for ROI {roi} on Test Set")
    # plt.xlabel("Z-score")
    # plt.ylabel("Frequency")
    # plt.show()

    # # Optionally: QQ plot to check normality
    # sm.qqplot(z_scores_test, line='s')
    # plt.title(f"QQ-plot of Z-scores for ROI {roi} on Test Set")
    # plt.show()


    # ## variation of residuals with predicted values
    # plt.scatter(y_pred, residuals, alpha=0.5)
    # plt.axhline(0, color='red', linestyle='--')
    # plt.xlabel("Predicted ROI")
    # plt.ylabel("Residuals")
    # plt.title("Residuals vs Predicted Values for "+roi)
    # plt.show()

    # # should follow a straight line (homogeneity of variance)
    # sm.qqplot(residuals, line='s')
    # plt.title("Q-Q Plot of Residuals")
    # plt.show()

    return pipeline, resid_std, epsilon_opt, beta_opt


def compute_zscores(roi_values, X_new, model, resid_std, warp=False, epsilon_opt=0, beta_opt=1):
    """
    Aim : function to compute z-score for new subjects
    """
    # Predict expected ROI values
    roi_pred = model.predict(X_new)
    if warp : roi_pred = inv_warp_sinh_arcsinh(roi_pred, epsilon=epsilon_opt, b=beta_opt)
    # Compute z-scores
    roi_values = np.asarray(roi_values)
    z_scores = (roi_values - roi_pred) / resid_std

    return z_scores

def plot_zscore_changes_by_group(index, m0_label0, m3_label0, m0_label1, m3_label1, 
                                label_names=["NR/PaR", "GR"]):
    """
    Plot paired z-score changes from z1 to z2 per participant for two groups,
    with lines connecting paired z1 and z2 scores, colored by group.
    
    Parameters:
    - m0_label0, m3_label0: arrays of z-scores for group 0
    - m0_label1, m3_label1: arrays of z-scores for group 1
    - label_names: list of two strings for legend labels
    """

    x_positions = [1, 2]
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['tab:blue', 'tab:orange']
    
    # Plot group 0
    for i in range(len(m0_label0)):
        ax.plot(x_positions, [m0_label0[i], m3_label0[i]], color=colors[0], alpha=0.5)
    ax.scatter(np.repeat(x_positions[0], len(m0_label0)), m0_label0, color=colors[0], label=f"{label_names[0]}: M0", alpha=0.8)
    ax.scatter(np.repeat(x_positions[1], len(m3_label0)), m3_label0, color=colors[0], marker='x', label=f"{label_names[0]}: M3", alpha=0.8)
    
    # Plot group 1
    for i in range(len(m0_label1)):
        ax.plot(x_positions, [m0_label1[i], m3_label1[i]], color=colors[1], alpha=0.5)
    ax.scatter(np.repeat(x_positions[0], len(m0_label1)), m0_label1, color=colors[1], label=f"{label_names[1]}: M0", alpha=0.8)
    ax.scatter(np.repeat(x_positions[1], len(m3_label1)), m3_label1, color=colors[1], marker='x', label=f"{label_names[1]}: M3", alpha=0.8)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['M0', 'M3'])
    ax.set_ylabel('Z-score')
    ax.set_title('Z-score changes from M0 to M3 by group in '+FOUR_REGIONS_OF_INTEREST_LONG[index])
    ax.legend()
    plt.show()

def scatter_M0_vs_M3_by_label(m0_by_label, m3_by_label, index,label_names=None, colors=None):
    """
    Scatter plot with M0 on x-axis and M3 on y-axis.
    Points are colored by label.

    Parameters:
    - m0_by_label: list of arrays, each containing M0 values for one label
    - m3_by_label: list of arrays, each containing M3 values for one label
    - label_names: optional list of label names for the legend
    - colors: optional list of colors to use for each label
    """
    assert len(m0_by_label) == len(m3_by_label), "Mismatch in number of labels"
    roi_rlink = FOUR_REGIONS_OF_INTEREST_LONG[index]

    n_labels = len(m0_by_label)

    assert label_names is not None
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_labels)]

    fig, ax = plt.subplots(figsize=(7, 7))

    all_vals = []
    for i in range(n_labels):
        ax.scatter(m0_by_label[i], m3_by_label[i], color=colors[i], alpha=0.7, label=label_names[i])
        all_vals.extend(m0_by_label[i])
        all_vals.extend(m3_by_label[i])

    # Identity line (x = y)
    all_vals = np.asarray(all_vals)
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='x = y')

    ax.set_xlabel("Z-score at M0",fontsize=18)
    ax.set_ylabel("Z-score at M3", fontsize=18)
    ax.set_title("M0 vs M3 Colored by Label "+roi_rlink, fontsize=20)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_violin_6_groups(zscoresM0_labelGR, zscoresM3_labelGR, zscoresM0_labelPaR, zscoresM3_labelPaR,
                                      zscoresM0_labelNR, zscoresM3_labelNR, index):
    # Flatten and build dataframe
    zscores = np.concatenate([
        zscoresM0_labelGR, zscoresM3_labelGR,
        zscoresM0_labelPaR, zscoresM3_labelPaR,
        zscoresM0_labelNR, zscoresM3_labelNR,
    ])
    roi_rlink = FOUR_REGIONS_OF_INTEREST_LONG[index]

    group_labels = (
        ['M0 GR'] * len(zscoresM0_labelGR) +
        ['M3 GR'] * len(zscoresM3_labelGR) +
        ['M0 PaR'] * len(zscoresM0_labelPaR) +
        ['M3 PaR'] * len(zscoresM3_labelPaR) +
        ['M0 NR'] * len(zscoresM0_labelNR) +
        ['M3 NR'] * len(zscoresM3_labelNR)
    )
    
    df = pd.DataFrame({'Z-score': zscores, 'Response': group_labels})
    
    # Define color palette: same color for M0/M3 of each label
    custom_palette = {
        'M0 GR': 'tab:blue',
        'M3 GR': 'tab:blue',
        'M0 PaR': 'tab:green',
        'M3 PaR': 'tab:green',
        'M0 NR': 'tab:orange',
        'M3 NR': 'tab:orange',
    }

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Response', y='Z-score', data=df, palette=custom_palette)

    plt.title("Z-score Distributions by Response and Timepoint "+roi_rlink, fontsize=18)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=30, fontsize=14)
    plt.xlabel("Response", fontsize=18)
    plt.ylabel("Z-score", fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_violin_4_groups(zscoresM0_label1, zscoresM3_label1,
                        zscoresM0_label0, zscoresM3_label0, index):
    
    # Flatten and build dataframe
    zscores = np.concatenate([
        zscoresM0_label1, zscoresM3_label1,
        zscoresM0_label0, zscoresM3_label0,
    ])
    roi_rlink = FOUR_REGIONS_OF_INTEREST_LONG[index]

    group_labels = (
        ['M0 GR'] * len(zscoresM0_label1) +
        ['M3 GR'] * len(zscoresM3_label1) +
        ['M0 NR/PaR'] * len(zscoresM0_label0) +
        ['M3 NR/PaR'] * len(zscoresM3_label0)
    )
    
    df = pd.DataFrame({'Z-score': zscores, 'Response': group_labels})
    
    # Define color palette: same color for M0/M3 of each label
    custom_palette = {
        'M0 GR': 'tab:blue',
        'M3 GR': 'tab:blue',
        'M0 NR/PaR': 'tab:orange',
        'M3 NR/PaR': 'tab:orange',
    }

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Response', y='Z-score', data=df, palette=custom_palette)

    plt.title("Z-score Distributions by Response and Timepoint "+roi_rlink, fontsize=18)

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=30, fontsize=14)
    plt.xlabel("Response", fontsize=18)
    plt.ylabel("Z-score", fontsize=18)

    plt.tight_layout()
    plt.show()

def get_zscores_by_label(label, index, warp=False, comparable_zscores_N91=True):
    """
        label (int) classification label
        index (int) index of roi in FOUR_REGIONS_OF_INTEREST and FOUR_REGIONS_OF_INTEREST_LONG lists
    """
    if label in [0,1]: label_type="int"
    if label in ["GR","PaR","NR"]: label_type="str"
    # here we compare the same subjects before and after lithium (by looking at their zscores)

    roi = FOUR_REGIONS_OF_INTEREST[index]
    roi_rlink = FOUR_REGIONS_OF_INTEREST_LONG[index]
    print(roi_rlink)
    model, resid_std, eps, beta = train_normative_model_with_openBHB(roi, warp=warp)
    
    df_M0, df_M3 = get_M0_M3_df_for_chosen_label(label, label_type, comparable_dataframes_N91=comparable_zscores_N91)
    roi_values_M0 = df_M0[roi_rlink].values
    roi_values_M3 = df_M3[roi_rlink].values

    X_rlink_M0 = df_M0[['age', 'sex']]
    X_rlink_M3 = df_M3[['age', 'sex']]

    z_scores_M0 = compute_zscores(roi_values_M0, X_rlink_M0, model, resid_std, warp=warp, epsilon_opt=eps, beta_opt=beta)
    z_scores_M3 = compute_zscores(roi_values_M3, X_rlink_M3, model, resid_std, warp=warp, epsilon_opt=eps, beta_opt=beta)
    print(f"Z-scores M0: mean={np.mean(z_scores_M0):.3f}, std={np.std(z_scores_M0, ddof=1):.3f}")
    print(f"Z-scores M0: min={np.min(z_scores_M0):.3f}, max={np.max(z_scores_M0):.3f}")

    print(f"Z-scores M3: mean={np.mean(z_scores_M3):.3f}, std={np.std(z_scores_M3, ddof=1):.3f}")
    print(f"Z-scores M3: min={np.min(z_scores_M3):.3f}, max={np.max(z_scores_M3):.3f}")

    if comparable_zscores_N91:
        closer_to_zero = np.abs(z_scores_M3) < np.abs(z_scores_M0)

        # Calculate percentage of participants of current label where z_scores_M3 is closer to zero
        percentage_z_scores_M3_closer = 100 * np.mean(closer_to_zero)

        print(f"Percentage of participants with zscores at M3 closer to zero than zscores at M0\
        : {percentage_z_scores_M3_closer:.2f}%")
        print("\n\n")

    return z_scores_M0, z_scores_M3

def print_participant_id_where_m0_is_over_m3(roi_index=0):
    """
    prints participant ids and correponding label for subjects for a specific roi
    where GM volume is higher at M0 than M3
    """
    roi_rlink = FOUR_REGIONS_OF_INTEREST_LONG[roi_index]
    print(roi_rlink)
    df = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    pivot_df = df.pivot(index='participant_id', columns='session', values=roi_rlink)
    
    # filter participant_ids where truc at M03 is less than at M00
    filtered_ids = pivot_df[pivot_df['M03'] < pivot_df['M00']].index
    y_values = df.drop_duplicates(subset='participant_id')[['participant_id', 'y']].set_index('participant_id')
    result = y_values.loc[filtered_ids]

    # for pid, y_val in result['y'].items():
    #     print(f"participant_id: {pid}, y: {y_val}")

    count_gr = sum(1 for _, y in result['y'].items() if y == 'GR')
    count_nr = sum(1 for _, y in result['y'].items() if y == 'NR')
    count_par = sum(1 for _, y in result['y'].items() if y == 'PaR')

    print(f"Number of 'GR' labels where volume is higher at baseline : {count_gr}")
    print(f"Number of 'NR' labels: {count_nr}")
    print(f"Number of 'PaR' labels: {count_par}")
    print(f"total: {count_par+count_nr+count_gr}/91")

def main():
    for index in range(4): print_participant_id_where_m0_is_over_m3(roi_index=index)

    quit()

    # train NM for each roi of the list
    # for index in range(4):
    #     roi = FOUR_REGIONS_OF_INTEREST[index]
    #     model, resid_std = train_normative_model_with_openBHB(roi)
    labels_onevsall = False
    comparable_zscores_N91=False
    for index in range(4):
        if labels_onevsall: # GR vs NR/PaR
            zscoresM0_label0, zscoresM3_label0 = get_zscores_by_label(0, index, comparable_zscores_N91=comparable_zscores_N91)
            zscoresM0_label1, zscoresM3_label1 = get_zscores_by_label(1, index, comparable_zscores_N91=comparable_zscores_N91)
            m0_by_label = [zscoresM0_label1, zscoresM0_label0]
            m3_by_label = [zscoresM3_label1, zscoresM3_label0]
            
            if comparable_zscores_N91: scatter_M0_vs_M3_by_label(m0_by_label, m3_by_label, index, label_names=["GR","PaR/NR"], colors=None)
            else: plot_violin_4_groups(zscoresM0_label1, zscoresM3_label1, zscoresM0_label0, zscoresM3_label0, index)

        else:  # case in which we want plots detailed for GR, NR, and PaR
            zscoresM0_labelGR, zscoresM3_labelGR = get_zscores_by_label("GR", index, comparable_zscores_N91=comparable_zscores_N91)
            zscoresM0_labelNR, zscoresM3_labelNR = get_zscores_by_label("NR", index, comparable_zscores_N91=comparable_zscores_N91)
            zscoresM0_labelPaR, zscoresM3_labelPaR = get_zscores_by_label("PaR", index, comparable_zscores_N91=comparable_zscores_N91)
            m0_by_label=[zscoresM0_labelGR, zscoresM0_labelPaR, zscoresM0_labelNR]
            m3_by_label = [zscoresM3_labelGR, zscoresM3_labelPaR, zscoresM3_labelNR]
            if comparable_zscores_N91:  scatter_M0_vs_M3_by_label(m0_by_label, m3_by_label,index, label_names=["GR","PaR","NR"], colors=None)
            else : plot_violin_6_groups(zscoresM0_labelGR, zscoresM3_labelGR, zscoresM0_labelPaR, zscoresM3_labelPaR, 
                            zscoresM0_labelNR, zscoresM3_labelNR, index)
        # plot_zscore_changes_by_group(index, zscoresM0_label0, zscoresM3_label0, zscoresM0_label1, zscoresM3_label1)

    

if __name__ == "__main__":
    main()
