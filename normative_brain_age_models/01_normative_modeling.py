import numpy as np
import pandas as pd
import sys, json, pickle, os
from normative_models import NormativeBLR

# to fit the normative model on OpenBHB data
from utils import stratified_split, get_lists_roi_in_both_openBHB_and_rlink, stratified_split_balanced_age

# to evaluate normative model
from sklearn.metrics import r2_score

# zscores classification
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#plots
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# residualization
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer
from nitk.ml_utils.residualization import get_residualizer

# inputs
ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
OPENBHB_DATAFRAME = DATA_DIR+"OpenBHB_roi.csv"
DF_PATH_HEALTHY_BRAINS = DATA_DIR + "hcp_open_mind_ukb_openbhb_roi_vbm_with_age.csv"

# RLink inputs
RLINK_DATAFRAME_M00_M03 = DATA_DIR + "df_ROI_age_sex_site_M00_M03_v4labels.csv"
RLINK_DATAFRAME_ALL_M00 = DATA_DIR + "df_ROI_age_sex_site_M00_v4labels.csv"
SELECTED_REGIONS_OF_INTEREST_RLINK = ['Left Hippocampus_GM_Vol','Right Hippocampus_GM_Vol','Left Amygdala_GM_Vol', 'Right Amygdala_GM_Vol']
CV_SPLITS=ROOT+"03_classif_rois/stratified-5cv.json"

# outputs
MODEL_DIR=ROOT+"models/normative_model/"
ZSCORES_PATH=ROOT+"reports/normative_modeling/"
# - roi_values has shape (n_subjects, n_rois)
# - covariates has shape (n_subjects, 2) for [age, sex]

# RLINK tests
def get_M0_M3():
    """
        returns two dataframes of ROIs at baseline (M0) and 3 months after li intake (M3) 
    """
    df_M0M3 = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    df_M0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    df_M3 = df_M0M3[df_M0M3["session"] == "M03"].copy()
    df_M3 = df_M3.reset_index(drop=True)
    df_M0 = df_M0.reset_index(drop=True)
    df_M0["response"] = df_M0["response"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_M3["response"] = df_M3["response"].replace({"GR": 1, "PaR": 0, "NR": 0})

    return df_M0, df_M3

def fit_normative_model(residualize=False, evaluate=False):
    # df_HC = pd.read_csv(OPENBHB_DATAFRAME)
    df_HC = pd.read_csv(DF_PATH_HEALTHY_BRAINS)

    df_train, df_test, train_idx, test_idxs  = stratified_split_balanced_age(df_HC, test_size=0.1) #stratified_split(df_HC, verbose=False)
    list_roi_openbhb, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    print("nb of openBHB rois :", len(list_roi_openbhb)," nb rois rlink ", len(list_roi_rlink))
    # assert len(df_HC)==len(df_train)+len(df_test)
    y_tr =  df_train[list_roi_openbhb].values
    y_te =  df_test[list_roi_openbhb].values


    if residualize: # residualizing on site only
        # print("openbhb train set before residualization:\n", y_tr)
        df_all_cov = pd.concat([df_train[["age","sex","site"]], df_test[["age","sex","site"]]], axis=0, ignore_index=True)
        assert len(df_all_cov)==len(df_train)+len(df_test)

        residualizer = Residualizer(data=df_all_cov, formula_res="site", formula_full="site + sex + age")
        Zres = residualizer.get_design_mat(df_all_cov)
        residualizer.fit(y_tr, Zres[:len(df_train)])
        
        y_tr = residualizer.transform(y_tr, Zres[:len(df_train)])
        y_te = residualizer.transform(y_te, Zres[len(df_train):])
        # print("openbhb train set after residualization:\n",y_tr)

    X_tr = df_train[['age', 'sex']]
    X_te = df_test[['age', 'sex']]
    print("Xtr , ytr",np.shape(X_tr), np.shape(y_tr))
    print("X_te , y_te",np.shape(X_te), np.shape(y_te))

    model = NormativeBLR(warp=True, bsplines=True)
    model.fit(X_tr, y_tr)

    # OPENBHB tests
    if evaluate: 

        # Predict means
        roi_pred_tr = model.predict(X_tr)
        print("r2 on train set ", r2_score(y_tr, roi_pred_tr))

        roi_pred = model.predict(X_te)
        # compute overall r2 metric on test set
        r2 = r2_score(y_te, roi_pred)
        print("r2 score on test set",r2)

        # Predict and get z-scores
        roi_pred, z_scores = model.predict(X_te, y_te, return_zscores=True)
        print("z_scores ",np.shape(z_scores), type(z_scores))

        df_yte = df_test[list_roi_openbhb]
        df_zscores_te = pd.DataFrame(z_scores, columns = list_roi_openbhb)

        corr_matrix_te = df_yte.corr()
        corr_matrix_te_zscores = df_zscores_te.corr()

        cond_number_te = np.linalg.cond(corr_matrix_te.values)
        fro_norm = np.linalg.norm(corr_matrix_te.values, ord='fro')
        print(f"Frobenius norm: {fro_norm:.3f}")
        print(f"Condition number: {cond_number_te:.2e}")

        cond_number_te_zscores = np.linalg.cond(corr_matrix_te_zscores.values)
        fro_norm = np.linalg.norm(corr_matrix_te_zscores.values, ord='fro')
        print(f"Frobenius norm: {fro_norm:.3f}")
        print(f"Condition number: {cond_number_te_zscores:.2e}")
    
    return model




def get_zscores(model, list_roi_rlink, df_M0, df_M3, residualize_on_site=False):
    # responses
    roi_values_M0 = df_M0[list_roi_rlink].values
    roi_values_M3 = df_M3[list_roi_rlink].values

    # covariates
    X_rlink_M0 = df_M0[['age', 'sex']]
    X_rlink_M3 = df_M3[['age', 'sex']]

    if residualize_on_site:
        residualizer_M0 = Residualizer(data=df_M0[["age","sex","site"]], formula_res="site", formula_full="site + sex + age")
        residualizer_M3 = Residualizer(data=df_M3[["age","sex","site"]], formula_res="site", formula_full="site + sex + age")

        ZresM0 = residualizer_M0.get_design_mat(df_M0[["age","sex","site"]])
        ZresM3 = residualizer_M3.get_design_mat(df_M3[["age","sex","site"]])

        residualizer_M0.fit(roi_values_M0, ZresM0)
        residualizer_M3.fit(roi_values_M3, ZresM3)

        roi_values_M0 = residualizer_M0.transform(roi_values_M0, ZresM0)
        roi_values_M3 = residualizer_M3.transform(roi_values_M3, ZresM3)

    roi_pred_M0, z_scores_M0 = model.predict(X_rlink_M0, roi_values_M0, return_zscores=True)
    roi_pred_M3, z_scores_M3 = model.predict(X_rlink_M3, roi_values_M3, return_zscores=True)

    df_zscores_M0 = pd.DataFrame(z_scores_M0, columns = list_roi_rlink)
    df_zscores_M0.insert(0, 'participant_id', df_M0["participant_id"])  
    df_zscores_M0["response"]=df_M0["response"]
    df_zscores_M3 = pd.DataFrame(z_scores_M3, columns = list_roi_rlink)
    df_zscores_M3.insert(0, 'participant_id', df_M3["participant_id"])  
    df_zscores_M3["response"]=df_M3["response"]
    print("zscores M0 ",df_zscores_M0)
    # print("zscores M3 ",df_zscores_M3)
    return df_zscores_M0, df_zscores_M3


def plot_normative_roi_by_age(df, roi_name, normative_model, 
                                age_col='age', class_col="response", sex_col='sex',
                                figsize=(12, 8)):
    """
    Plot z-scores from normative model by age, with healthy normative curve.
    
    Parameters:
    -----------
    df_zscores : pandas.DataFrame
        DataFrame containing age, sex, class labels, and pre-computed z-scores
    roi_name : str
        Name of the ROI z-score column to plot
    normative_model : NormativeBLR
        Fitted normative model (used to plot healthy normative curve)
    """
    
    if roi_name not in df.columns:
        raise ValueError(f"ROI '{roi_name}' not found in columns")
    
    # Extract bipolar subject data
    ages = df[age_col].values
    zscores = df[roi_name].values
    classes = df[class_col].values
    
    # Separate by class
    class_0_mask = classes == 0
    class_1_mask = classes == 1
    
    ages_0 = ages[class_0_mask]
    ages_1 = ages[class_1_mask]
    zscores_0 = zscores[class_0_mask]
    zscores_1 = zscores[class_1_mask]
    
    # Create age range for plotting normative curve
    age_min, age_max = ages.min(), ages.max()
    age_range = np.linspace(age_min, age_max, 100)
    
    # Get normative predictions for both sexes
    roi_cols = [col for col in df.columns 
                if col not in [age_col, sex_col, class_col, 'participant_id']]
    roi_idx = roi_cols.index(roi_name.replace('_zscore', '')) if roi_name.replace('_zscore', '') in roi_cols else 0
    
    # Plot normative curves for both sexes
    fig, ax = plt.subplots(figsize=figsize)
    
    for sex_val in ["male", "female"]:
        X_norm = pd.DataFrame({
            'age': age_range,
            'sex': [sex_val] * len(age_range)
        })
        # X_norm = np.column_stack([age_range, np.full(len(age_range), sex_val)])
        predictions_norm = normative_model.predict(X_norm)
        
        sex_label = 'Female' if sex_val == 0 else 'Male'
        color = 'pink' if sex_val == 0 else 'lightblue'
        ax.plot(age_range, predictions_norm[:, roi_idx], 
                color=color, alpha=0.7, linewidth=2, 
                label=f'Healthy {sex_label} mean')
        
        # Add confidence bands (±1 std)
        std_val = normative_model.stds_[roi_idx]
        ax.fill_between(age_range, 
                       predictions_norm[:, roi_idx] - std_val,
                       predictions_norm[:, roi_idx] + std_val,
                       color=color, alpha=0.2)
    
    # Plot bipolar subject z-scores
    ax.scatter(ages_0, zscores_0, alpha=0.7, s=50, 
              color='blue', label=f'Bipolar Class 0 (n={len(ages_0)})')
    ax.scatter(ages_1, zscores_1, alpha=0.7, s=50, 
              color='red', label=f'Bipolar Class 1 (n={len(ages_1)})')
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    ax.axhline(y=1.96, color='gray', linestyle='--', alpha=0.7, label='±1.96σ')
    ax.axhline(y=-1.96, color='gray', linestyle='--', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Brain measure / Z-score', fontsize=12)
    ax.set_title(f'Normative Model: {roi_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_normative_zscore_by_age(df_zscores, roi_name, 
                                 figsize=(12, 8)):
    """
    Plot z-scores from normative model by age, colored by class label.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing subjects with age, class labels, and ROI data
    roi_name : str
        Name of the ROI column to plot
    figsize : tuple
        Figure size (default: (12, 8))
    """
    
    # Check if ROI exists in dataframe
    if roi_name not in df_zscores.columns:
        raise ValueError(f"ROI '{roi_name}' not found in columns")
    
    # Extract data
    ages = df_zscores['age'].values
    zscores = df_zscores[roi_name].values
    classes = df_zscores["response"].values
    
    # Separate by class
    class_0_mask = classes == 0
    class_1_mask = classes == 1
    
    ages_0 = ages[class_0_mask]
    ages_1 = ages[class_1_mask]
    zscores_0 = zscores[class_0_mask]
    zscores_1 = zscores[class_1_mask]

    palette = sns.color_palette()
    fig, ax = plt.subplots(figsize=figsize)
    
    lowess_0 = lowess(zscores_0, ages_0, frac=0.3)
    ax.plot(lowess_0[:, 0], lowess_0[:, 1], color=palette[0], 
            linewidth=3, label=f'Non responders (n={len(ages_0)})')
    
    # Optional: Add scatter points with lower alpha
    ax.scatter(ages_0, zscores_0, alpha=0.3, s=30, color=palette[0])

    lowess_1 = lowess(zscores_1, ages_1, frac=0.3)
    ax.plot(lowess_1[:, 0], lowess_1[:, 1], color=palette[1], 
            linewidth=3, label=f'Responders (including partial responders) (n={len(ages_1)})')
    
    # Optional: Add scatter points with lower alpha
    ax.scatter(ages_1, zscores_1, alpha=0.3, s=30, color=palette[1])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.9, linewidth=2.5)
    ax.axhline(y=1.96, color='gray', linestyle='--', alpha=0.7, label='±1.96σ')
    ax.axhline(y=-1.96, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.set_title(f'Deviation from normative "healthy" gray matter (Z-score = 0) in : {roi_name[:-len("_GM_Vol")]}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_normative_roi_by_age_using_zscores(df_zscores, roi_name, normative_model,
                             age_col='age', class_col="response", sex_col='sex',
                             figsize=(12, 12), lowess_frac=0.3):
    """
    Plot original ROI values converted from z-scores, with separate plots for male/female.
    
    Parameters:
    -----------
    df_zscores : pandas.DataFrame
        DataFrame containing age, sex, class labels
    roi_name : str
        Name of the ROI z-score column to plot
    normative_model : NormativeBLR
        Fitted normative model
    lowess_frac : float
        Fraction of data used for LOWESS smoothing
    """
    
    if roi_name not in df_zscores.columns:
        raise ValueError(f"ROI '{roi_name}' not found in columns")
    
    # Get ROI index
    roi_cols = [col for col in df_zscores.columns 
                if col not in [age_col, sex_col, class_col, 'participant_id']]
    roi_idx = roi_cols.index(roi_name)
    
    # Get model predictions for all subjects
    X_covariates = df_zscores[[age_col, sex_col]].values
    predictions = normative_model.predict(X_covariates)
    residual_std = normative_model.stds_[roi_idx]
    
    # Convert z-scores back to original GM volumes
    zscores = df_zscores[roi_name].values
    original_volumes = (zscores * residual_std) + predictions[:, roi_idx]
    
    # Extract data
    ages = df_zscores[age_col].values
    classes = df_zscores[class_col].values
    sexes = df_zscores[sex_col].values
    
    # Get seaborn colors
    colors = sns.color_palette()
    blue_color = colors[0]    # Non-responders
    orange_color = colors[1]  # Responders
    
    # Create subplots: Female on top (sex=1), Male below (sex=0)
    fig, (ax_female, ax_male) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Age range for normative curves
    age_min, age_max = ages.min(), ages.max()
    age_range = np.linspace(age_min, age_max, 100)
    
    for sex_val, ax, sex_title in [("female", ax_female, 'Female'), ("male", ax_male, 'Male')]:
        # Filter data for this sex
        sex_mask = sexes == sex_val
        X_norm = pd.DataFrame({
            'age': age_range,
            'sex': [sex_val] * len(age_range)
        })
        
        predictions_norm = normative_model.predict(X_norm)
        # print("sex_val ",sex_val)
        # print("X_norm ",X_norm)
        # print("predictions_norm[:, roi_idx] ",predictions_norm[:, roi_idx])
        
        # Plot normative curve in grey
        ax.plot(age_range, predictions_norm[:, roi_idx], 
                color='grey', alpha=0.8, linewidth=2, 
                label='Healthy mean')
        
        # Add confidence bands (±1 std) in grey
        ax.fill_between(age_range, 
                       predictions_norm[:, roi_idx] - residual_std,
                       predictions_norm[:, roi_idx] + residual_std,
                       color='grey', alpha=0.2, label='±1 SD')
        
        # Get data for this sex
        sex_ages = ages[sex_mask]
        sex_volumes = original_volumes[sex_mask]
        sex_classes = classes[sex_mask]
        
        # Separate by response class within this sex
        class_0_mask_sex = sex_classes == 0
        class_1_mask_sex = sex_classes == 1
        
        ages_0_sex = sex_ages[class_0_mask_sex]
        ages_1_sex = sex_ages[class_1_mask_sex]
        volumes_0_sex = sex_volumes[class_0_mask_sex]
        volumes_1_sex = sex_volumes[class_1_mask_sex]
        
        # Plot scatter points
        if len(ages_0_sex) > 0:
            ax.scatter(ages_0_sex, volumes_0_sex, alpha=0.6, s=40,
                      color=blue_color, label=f'Non-responders (n={len(ages_0_sex)})')
        
        if len(ages_1_sex) > 0:
            ax.scatter(ages_1_sex, volumes_1_sex, alpha=0.6, s=40,
                      color=orange_color, label=f'Good responders (n={len(ages_1_sex)})')
        
        # Add LOWESS curves
        if len(ages_0_sex) > 3:
            lowess_0 = lowess(volumes_0_sex, ages_0_sex, frac=lowess_frac)
            ax.plot(lowess_0[:, 0], lowess_0[:, 1], color=blue_color, 
                    linewidth=3, alpha=0.8)
        
        if len(ages_1_sex) > 3:
            lowess_1 = lowess(volumes_1_sex, ages_1_sex, frac=lowess_frac)
            ax.plot(lowess_1[:, 0], lowess_1[:, 1], color=orange_color, 
                    linewidth=3, alpha=0.8)
        
        # Formatting for each subplot
        ax.set_ylabel('Gray Matter Volume', fontsize=12)
        ax.set_title(f'{sex_title}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # Set x-label only for bottom plot
    ax_male.set_xlabel('Age (years)', fontsize=12)
    
    # Overall title
    fig.suptitle(f'Normative Model: {roi_name[:-len("_GM_Vol")]}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.show()
    return fig, (ax_female, ax_male)

def plot_age_distrib_resp_or_noresp(df, y):
    """
    generates bar plot of age distribution for responders (y=1) or non responders (y=0)
    bars are colored by sex (male = 0, female = 1)
    """
    # Filter data where y == 0
    df_filtered = df[df["response"] == y]

    # Create bar plot of age distribution colored by age
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_filtered, x='age', hue='sex')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title(f'Age Distribution for y={y}, colored by Sex')
    plt.legend(title='Sex', labels=['Male (0)', 'Female (1)'])
    plt.xticks(rotation=45)
    plt.show()


def classification_from_zscores(model, scale=False, residualize_on_site=False):
    with open(CV_SPLITS, "r") as f:
        splits = json.load(f)

    _, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    df_M0, df_M3 = get_M0_M3()
    df, _ = get_zscores(model, list_roi_rlink, df_M0, df_M3) 
    # warning : in get_zscores: do NOT apply residualization to all zscores at once (data leakage for classification)

    assert max(x for outer in splits for inner in outer for x in inner) == 116
    X = df.drop(columns=["response","participant_id"]).values
    y = df["response"].values
    # Define model
    clf = lm.LogisticRegression(fit_intercept=True, class_weight='balanced')
    # clf = GridSearchCV(
    #         lm.LogisticRegression(class_weight="balanced", fit_intercept=False),
    #         param_grid={"C": [0.1, 1.0, 10.0]},
    #         cv=3,            
    #         scoring="roc_auc",  
    #         n_jobs=-1
    #     )

    scaler = StandardScaler()
    scores = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if residualize_on_site:
            residualizer = Residualizer(data=df_M0[["age","sex","site"]], formula_res="site", formula_full="site + sex + age")
            Zres = residualizer.get_design_mat(df_M0[["age","sex","site"]])
            assert len(Zres)==len(X)
            residualizer.fit(X_train, Zres[train_idx])
            X_train = residualizer.transform(X_train, Zres[train_idx])
            X_test = residualizer.transform(X_test, Zres[test_idx])

        if scale:
            # Scale features
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit model
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        scores.append({"fold": fold, "accuracy": acc, "roc_auc": auc})

    # Convert results to DataFrame
    df_scores = pd.DataFrame(scores)
    print(df_scores)
    print("Mean AUC:", df_scores["roc_auc"].mean())

def get_roi_diff_percentages(df0, df3, list_roi):
    """
    df0 : dataframe with M0 values
    df3 : dataframe with M3 values
    list_roi : list of rois
    """
    assert df3["participant_id"].is_unique

    # Keep only participants with M0 and M3 measures
    df0_common = df0[df0["participant_id"].isin(df3["participant_id"])]
    assert df0_common["participant_id"].is_unique

    # Align on participant_id to ensure matching order
    df0_common = df0_common.set_index("participant_id").loc[df3["participant_id"].values.tolist()].reset_index()
    df3_common = df3.reset_index(drop=True)
    # print(df0_common)
    # print(df3_common)

    assert all(df0_common["participant_id"] == df3_common["participant_id"])
    assert all(df0_common["response"] == df3_common["response"])

    results = {}

    for label in [0, 1]:
        # Filter rows where y == label
        mask = df0_common["response"] == label
        d0 = df0_common.loc[mask, list_roi].abs()
        d3 = df3_common.loc[mask, list_roi].abs()

        # Compute per-ROI % of participants where |df3| < |df0|
        # --> equivalent to a closening to the "norm" after Li intake
        percentages = (d3 < d0).sum(axis=0) / len(d0) * 100
        results[label] = round(percentages,2)

    return results  # dict: {0: Series, 1: Series}

def save_zscores(residualize_on_site=False):
    """
    1. fit normative model if it doesn't already exist
    2. save df of zscores for either
        A. a normative model trained on Healthy Controls ROI residualized (on site),
            and zscores estimated from RLINK ROI residualized (on site)
        B. a normative model trained on Healthy Controls ROI and zscores estimates from RLINK ROI
    """
    if residualize_on_site: modelname = "normative_BLR_BigHC_site_residualized"
    else : modelname = "normative_BLR_BigHC" 
    model_path = MODEL_DIR+modelname+".pkl"
    if not os.path.exists(model_path):
        model = fit_normative_model(residualize=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    _, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    df_M0, df_M3 = get_M0_M3()
    df_zscores_M0, df_zscores_M3 = get_zscores(model, list_roi_rlink, df_M0, df_M3, residualize_on_site=residualize_on_site)
    # classification_from_zscores(model, scale=True, residualize_on_site=True)

    # plot_age_distrib_resp_or_noresp(df_M0, y=1)
    df_zscores_M0_with_age_sex_site = pd.merge(df_zscores_M0, df_M0[["participant_id","age","sex","site"]],on="participant_id", how="inner")
    df_zscores_M3_with_age_sex_site  = pd.merge(df_zscores_M3, df_M0[["participant_id","age","sex","site"]],on="participant_id", how="inner")
    
    # save dataframes of zscores to csv
    df_zscores_M0_with_age_sex_site.to_csv(ZSCORES_PATH+"df_zscores_M0_with_age_sex_site_"+modelname+".csv", index=False)
    df_zscores_M3_with_age_sex_site.to_csv(ZSCORES_PATH+"df_zscores_M3_with_age_sex_site_"+modelname+".csv", index=False)

    # for i in range(4):
    #     plot_normative_zscore_by_age(df_zscores_M0_with_age_sex_site, SELECTED_REGIONS_OF_INTEREST_RLINK[i])
    print("df_M0 \n",df_M0[SELECTED_REGIONS_OF_INTEREST_RLINK])
    print("M3 GR zscores :\n", df_zscores_M3[df_zscores_M3["response"]==1][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())
    print("M3 NR/PaR zscores :\n",df_zscores_M3[df_zscores_M3["response"]==0][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())

    print("M0 GR zscores :\n", df_zscores_M0[df_zscores_M0["response"]==1][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())
    print("M0 NR/PaR zscores :\n",df_zscores_M0[df_zscores_M0["response"]==0][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())

def zscores_m3minusm0(residualize_on_site=False):
    """
    Saves a df of zscores M3 - zscores M0 ROI measures
        save_m3_minus_m0_df (bool) : 
            if True, save df of differences between zscores of m3 and zscores of m0 to csv. if False, don't.
        residualize_on_site: when the NM has been trained on ROI residualized on site, and the zscores have been estimated
            from ROI residualized on site as well (warning : the sites are different when estimating the NM on healthy controls
            from the sites of the dataset used to estimate zscores)
    """
    if residualize_on_site:
        modelname = "normative_BLR_BigHC_site_residualized"
        new_zscores_df_file = f"df_zscores_M3minusM0_with_age_sex_site_res_on_site_{modelname}.csv"

    else : 
        modelname = "normative_BLR_BigHC" 
        new_zscores_df_file = f"df_zscores_M3minusM0_with_age_sex_site_{modelname}.csv"

    _, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()

    path_df_M3minusM0 = ZSCORES_PATH+new_zscores_df_file

    if not os.path.exists(path_df_M3minusM0):
        df_ROI_age_sex_site_M0 = pd.read_csv(ZSCORES_PATH+f"df_zscores_M0_with_age_sex_site_{modelname}.csv")
        df_ROI_age_sex_site_M3 = pd.read_csv(ZSCORES_PATH+f"df_zscores_M3_with_age_sex_site_{modelname}.csv")
        print(df_ROI_age_sex_site_M3)
        print(df_ROI_age_sex_site_M0)
        # Merge M03 and M00 on participant_id
        merged = pd.merge(df_ROI_age_sex_site_M3, df_ROI_age_sex_site_M0, on="participant_id", suffixes=("_M03", "_M00"))

        list_ = ["response","age","sex","site"]
        for l in list_:
            assert (merged[l+'_M00'] == merged[l+'_M03']).all(), " issue with "+l+" between same subjects at M00 and M03"

        columns_M03 = [col for col in merged.columns if col.endswith("_M03")]
        columns_M00 = [col for col in merged.columns if col.endswith("_M00")]
        common_columns = [col[:-4] for col in columns_M03 if col[:-4] + "_M00" in columns_M00]
        print(len(list_roi_rlink))
        print("common_columns ", len(common_columns)) # == 273 since there are 268 ROIs (134 GM and 134 CSF) and response (label), age, sex, site, session
        print(merged)

        # creating a df for differences of M03-M00
        differences_df = pd.DataFrame({
            col: merged[col + "_M03"] - merged[col + "_M00"] for col in common_columns if not col in \
                ["participant_id", "age","sex","site","response","session"]
        })
        differences_df.insert(0, 'participant_id', merged['participant_id'])
        differences_df["response"] = merged["response_M00"]
        differences_df["age"] = merged["age_M00"]
        differences_df["sex"] = merged["sex_M00"]
        differences_df["site"] = merged["site_M00"]

        print(differences_df)
        differences_df.to_csv(path_df_M3minusM0,index=False)  

    else : differences_df = pd.read_csv(path_df_M3minusM0)

    print("differences grouped by label\n",differences_df[list_roi_rlink+["response"]].groupby("response").median())
    differences_df = differences_df.drop("participant_id", axis=1)
    print(differences_df)



# add violin plots here
def main():
    # save_zscores(True)
    zscores_m3minusm0(residualize_on_site=True)
    zscores_m3minusm0(residualize_on_site=False)

    quit()

    # plot_normative_roi_by_age_using_zscores(df_M0, SELECTED_REGIONS_OF_INTEREST_RLINK[1], model)
    for i in range(4):
        plot_normative_roi_by_age_using_zscores(df_zscores_M0_with_age_sex_site, SELECTED_REGIONS_OF_INTEREST_RLINK[i], model)
        # plot_normative_roi_by_age_using_zscores(df_zscores_M3_with_age_sex_site, SELECTED_REGIONS_OF_INTEREST_RLINK[i], model)
        
    quit()

   
    
    X_arr_res, residualizer_estimator, residualization_formula = \
                get_residualizer(df_M0, df_M0[list_roi_rlink].values, residualization_columns=["site"])
    print("df_M0 before res :",df_M0[list_roi_rlink])
    df_M0[list_roi_rlink] = residualizer_estimator.fit_transform(X_arr_res)
    print("df_M0 after res :",df_M0[list_roi_rlink])

  
    # print("correlation matrix Frobenius norm residualized M0 roi "\
    #         ,np.linalg.norm(df_roi_M0_res.corr().values, ord='fro'))
    plot_normative_roi_by_age(df_M0, SELECTED_REGIONS_OF_INTEREST_RLINK[1], model, 
                                age_col='age', class_col="response", sex_col='sex',
                                figsize=(12, 8))
    quit()

    
    # results = get_roi_diff_percentages(df_zscores_M0, df_zscores_M3, list_roi_rlink)
    print(df_zscores_M0)
    print(df_zscores_M3)
    for roi in SELECTED_REGIONS_OF_INTEREST_RLINK:
        print(roi)
        results = get_roi_diff_percentages(df_zscores_M0, df_zscores_M3, [roi])
        print(results)


    

if __name__ == "__main__":
    main()

"""

results of classification_from_zscores(df_zscores_M0, scale=True)

with normative_BLR_BigHC
classification from zscores with standard scaling:
    fold  accuracy   roc_auc
0     0  0.608696  0.611111
1     1  0.478261  0.525000
2     2  0.652174  0.750000
3     3  0.625000  0.688889
4     4  0.666667  0.696296
Mean AUC: 0.6542592592592593

with normative_BLR_BigHC
classification from zscores with standard scaling and residualization on site:
   fold  accuracy   roc_auc
0     0  0.695652  0.730159
1     1  0.608696  0.641667
2     2  0.826087  0.800000
3     3  0.666667  0.592593
4     4  0.583333  0.681481
Mean AUC: 0.6891798941798941

with normative_BLR_BigHC_site_residualized
classification from zscores with standard scaling and residualization on site:
   fold  accuracy   roc_auc
0     0  0.608696  0.690476
1     1  0.565217  0.666667
2     2  0.695652  0.675000
3     3  0.666667  0.570370
4     4  0.750000  0.703704
Mean AUC: 0.6612433862433863


with normative_BLR_OpenBHB
classification from zscores without standard scaling:
fold  accuracy   roc_auc
0     0  0.608696  0.587302
1     1  0.608696  0.516667
2     2  0.521739  0.541667
3     3  0.583333  0.666667
4     4  0.541667  0.607407
Mean AUC: 0.5839417989417989

classification from zscores with standard scaling:
    fold  accuracy   roc_auc
0     0  0.608696  0.626984
1     1  0.521739  0.566667
2     2  0.652174  0.741667
3     3  0.666667  0.651852
4     4  0.666667  0.659259
Mean AUC: 0.6492857142857142

classification with standard scaling and model trained with residualized (on site) roi
    fold  accuracy   roc_auc
0     0  0.565217  0.650794
1     1  0.521739  0.516667
2     2  0.652174  0.733333
3     3  0.666667  0.651852
4     4  0.625000  0.644444
Mean AUC: 0.6394179894179894

"""