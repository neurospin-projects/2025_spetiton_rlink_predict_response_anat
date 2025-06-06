import pandas as pd
import numpy as np
import sys
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
RLINK_DATAFRAME = DATA_DIR + "df_ROI_age_sex_site_M00_M03_v4labels.csv"



def get_M0_M3_df_for_chosen_label(label_value):
    assert label_value ==1 or label_value ==0," wrong label "

    df = pd.read_csv(RLINK_DATAFRAME)
    df["y"] = df["y"].replace({"GR": 1, "PaR": 0, "NR": 0})

    # get participant ids matching the given label (0=GR, 1=PaR/NR)
    participant_ids = df.loc[df["y"] == label_value, "participant_id"]

    # filder df by session to get m0 and m3 dataframes
    def filter_session(session_name):
        df_sess = df[df["session"] == session_name]
        df_sess = df_sess[df_sess["participant_id"].isin(participant_ids)].reset_index(drop=True)
        return df_sess

    df_M3 = filter_session("M03")
    df_M0 = filter_session("M00")
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


def stratified_split(df, stratify_cols=['age', 'sex', 'site'], test_size=0.2, random_state=42, verbose=False):
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


def train_normative_model_with_openBHB(roi, bsplines=True):
    df_openbhb = pd.read_csv(OPENBHB_DATAFRAME)

    # print("dataframe OpenBHB rois ...\n",df_openbhb)
    df_train, df_test = stratified_split(df_openbhb)

    # plot_spline_knots_effect(df_train, roi, sex_value=0)
    # plot_spline_knots_effect(df_train, roi, sex_value=1)

    # Create spline basis for age with 3 degrees of freedom (knots)
    # This creates new spline basis columns for age
    X_tr = df_train[['age', 'sex']]
    X_te = df_test[['age', 'sex']]
    y_tr =  df_train[roi]
    y_te =  df_test[roi]

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

    
    return pipeline, resid_std


def compute_zscores(roi_values, X_new, model, resid_std):
    """
    Aim : function to compute z-score for new subjects
    """
    # Predict expected ROI values
    roi_pred = model.predict(X_new)

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

def get_zscores_by_label(label, index):
    """
        label (int) classification label
        index (int) index of roi in FOUR_REGIONS_OF_INTEREST and FOUR_REGIONS_OF_INTEREST_LONG lists
    """
    # here we compare the same subjects before and after lithium (by looking at their zscores)

    roi = FOUR_REGIONS_OF_INTEREST[index]
    roi_rlink = FOUR_REGIONS_OF_INTEREST_LONG[index]
    print(roi_rlink)
    model, resid_std = train_normative_model_with_openBHB(roi)
    df_M0, df_M3 = get_M0_M3_df_for_chosen_label(label)
    roi_values_M0 = df_M0[roi_rlink].values
    roi_values_M3 = df_M3[roi_rlink].values

    X_rlink_M0 = df_M0[['age', 'sex']]
    X_rlink_M3 = df_M3[['age', 'sex']]

    z_scores_M0 = compute_zscores(roi_values_M0, X_rlink_M0, model, resid_std)
    z_scores_M3 = compute_zscores(roi_values_M3, X_rlink_M3, model, resid_std)
    print(f"Z-scores M0: mean={np.mean(z_scores_M0):.3f}, std={np.std(z_scores_M0, ddof=1):.3f}")
    print(f"Z-scores M0: min={np.min(z_scores_M0):.3f}, max={np.max(z_scores_M0):.3f}")

    print(f"Z-scores M3: mean={np.mean(z_scores_M3):.3f}, std={np.std(z_scores_M3, ddof=1):.3f}")
    print(f"Z-scores M3: min={np.min(z_scores_M3):.3f}, max={np.max(z_scores_M3):.3f}")

    closer_to_zero = np.abs(z_scores_M3) < np.abs(z_scores_M0)

    # Calculate percentage where z_scores_M3 is closer to zero
    percentage_z_scores_M3_closer = 100 * np.mean(closer_to_zero)

    print(f"Percentage of participants with zscores at M3 closer to zero than zscores at M0\
    : {percentage_z_scores_M3_closer:.2f}%")
    print("\n\n")
    return z_scores_M0, z_scores_M3

def main():
    for index in range(4):
        zscoresM0_label0, zscoresM3_label0 = get_zscores_by_label(0, index)
        zscoresM0_label1, zscoresM3_label1 = get_zscores_by_label(1, index)
        # plot_zscore_changes_by_group(index, zscoresM0_label0, zscoresM3_label0, zscoresM0_label1, zscoresM3_label1)

    

if __name__ == "__main__":
    main()
