import pandas as pd
import numpy as np
import sys, pathlib
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from utils import stratified_split, get_lists_roi_in_both_openBHB_and_rlink, stratified_split_balanced_age
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as lm
from sklearn.model_selection import StratifiedKFold
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer


# inputs
FOUR_REGIONS_OF_INTEREST = ["lHip_GM_Vol","rHip_GM_Vol","lAmy_GM_Vol","rAmy_GM_Vol"]
FOUR_REGIONS_OF_INTEREST_LONG = ['Left Hippocampus_GM_Vol','Right Hippocampus_GM_Vol','Left Amygdala_GM_Vol', 'Right Amygdala_GM_Vol']
ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
PATH_TO_DATA_OPENBHB = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/"
OPENBHB_DATAFRAME = DATA_DIR+"OpenBHB_roi.csv"
DF_PATH_HEALTHY_BRAINS = DATA_DIR + "hcp_open_mind_ukb_openbhb_roi_vbm_with_age.csv"
RLINK_DATAFRAME_M00_M03 = DATA_DIR + "df_ROI_age_sex_site_M00_M03_v4labels.csv"
RLINK_DATAFRAME_ALL_M00 = DATA_DIR + "df_ROI_age_sex_site_M00_v4labels.csv"

def group_response(label):
    if label == "GR":
        return "GR"
    elif label in ["PaR", "NR"]:
        return "PaR/NR"
    else:
        return label 

def residualize_data(df, X_arr_tr, tr = None, X_arr_te = None, te = None, \
                                 formula_res = "site + sex", formula_full="site + sex + age"):
    # define residualizer
    residualizer = Residualizer(data=df, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(df)
    # fit residualizer
    if tr is not None: Zres_tr = Zres[tr]
    else : Zres_tr = Zres
    residualizer.fit(X_arr_tr, Zres_tr)
    # apply residualizer
    X_arr_tr = residualizer.transform(X_arr_tr, Zres_tr)
    if te is not None and X_arr_te is not None:
        X_arr_te = residualizer.transform(X_arr_te, Zres[te])
    
    if X_arr_te is not None:
        return X_arr_tr, X_arr_te
    else:
        return X_arr_tr


def plot_age_distribution(df, cohort_name):
    """
    Plot the age distribution of the 'age' column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing an 'age' column.
    """
    if "age" not in df.columns:
        raise ValueError("The DataFrame must contain an 'age' column.")

    plt.figure(figsize=(8, 5))
    plt.hist(df["age"].dropna(), bins=30, edgecolor="black", alpha=0.7)
    plt.title(cohort_name+" age distribution", fontsize=14)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

def classify_site_rlink(model="EN"):
    df_M0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)

    if model=="EN":
        site_classification_model = GridSearchCV(
            estimator=ElasticNet(max_iter=10000),
            param_grid={
                "alpha": [0.01, 0.1, 1, 10],
                "l1_ratio": [0.1, 0.5, 0.9]  
            },
            cv=5,
            n_jobs=-1,
            scoring='balanced_accuracy',
            verbose=1
        )
    if model=="LR":
        site_classification_model = LogisticRegression()

    if model =="MLP":
        mlp_param_grid = {
            "hidden_layer_sizes": [
                (100,), (50,), (25,), (10,), (5,),  
                (100, 50), (50, 25), (25, 10), (10, 5),  
                (100, 50, 25), (50, 25, 10), (25, 10, 5)  
            ],
            "activation": ["relu"],
            "solver": ["sgd"],
            "alpha": [0.0001]
        }

        site_classification_model = GridSearchCV(
            estimator=MLPRegressor(random_state=1, max_iter=1000),
            param_grid=mlp_param_grid,
            cv=3,
            n_jobs=1,
            scoring="balanced_accuracy"
        )

    pipeline = make_pipeline(
        StandardScaler(), 
        site_classification_model)
    
    list_roi_openbhb, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    X_arr = df_M0[list_roi_rlink].values
    y_arr =  df_M0['site'].values 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean_bacc = []
    for fold_idx, (tr, te) in enumerate(skf.split(X_arr, y_arr)):
        X_arr_tr, X_arr_te = X_arr[tr], X_arr[te]
     
        pipeline.fit(X_arr_tr, y_arr[tr])
        y_pred = pipeline.predict(X_arr_te)
        bacc = balanced_accuracy_score(y_arr[te], y_pred)
        print("Balanced accuracy for OpenBHB test set:", round(bacc,2))
        mean_bacc.append(bacc)

    print("mean balanced accuracy ",np.mean(bacc))
    
def scale_x_y_agebins(X_train, X_test, df_train, df_test, age_bins):
    """
    Scale X_train / X_test per age bin using StandardScaler.

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
        Features for training subjects.
    X_test : array-like of shape (n_test, n_features)
        Features for test subjects.
    df_train : pd.DataFrame, must contain column "age"
    df_test : pd.DataFrame, must contain column "age"
    age_bins : pd.Series
        Age bins for all subjects (index aligned with df_train and df_test).
    """

    X_train_scaled = np.empty_like(X_train, dtype=float)
    X_test_scaled = np.empty_like(X_test, dtype=float)

    # Map bins to train/test indices
    age_bins_train = age_bins.loc[df_train.index]
    age_bins_test = age_bins.loc[df_test.index]

    scalers = {}

    # --- fit a scaler for each bin present in the training set ---
    for bin_label in age_bins_train.unique():
        idx_train = age_bins_train == bin_label
        scaler = StandardScaler()
        X_train_scaled[idx_train, :] = scaler.fit_transform(X_train[idx_train, :])
        scalers[bin_label] = scaler

    # also keep a fallback scaler trained on the whole training set
    global_scaler = StandardScaler().fit(X_train)
    scalers["__global__"] = global_scaler

    # --- apply to the test set ---
    for bin_label in age_bins_test.unique():
        idx_test = age_bins_test == bin_label
        if bin_label in scalers:
            X_test_scaled[idx_test, :] = scalers[bin_label].transform(X_test[idx_test, :])
        else:
            # use fallback if bin has no scaler (no training subjects)
            X_test_scaled[idx_test, :] = scalers["__global__"].transform(X_test[idx_test, :])

    return X_train_scaled, X_test_scaled, scalers

def scale_x_y_agebins_rlink(X_arr, df, train_bins, scalers):
    """
    Scale data per age bin using pre-fitted scalers from scale_x_y_agebins.

    params
    ----------
    X_arr : ndarray (n_subjects, n_features)
        Features to scale.
    df : pd.DataFrame
        Must contain column "age" for the new subjects.
    train_bins : pd.Series (from training)
        Age-bin series used during fit (to recover bin intervals).
    scalers : dict
        Mapping {bin_interval: fitted StandardScaler}.
    """
    # Get bin intervals from training
    intervals = train_bins.cat.categories

    # Bin ages of the new dataframe using the same intervals
    new_bins = pd.cut(df["age"], bins=[i.left for i in intervals] + [intervals[-1].right], include_lowest=True)

    X_scaled = np.empty_like(X_arr, dtype=float)

    for bin_label in new_bins.unique():
        idx = new_bins == bin_label
        if bin_label in scalers:
            X_scaled[idx, :] = scalers[bin_label].transform(X_arr[idx, :])
        else:
            # if a bin wasn't seen during training, fall back to global scaler or warning
            X_scaled[idx, :] = scalers["__global__"].transform(X_arr[idx])

    return X_scaled


def train_brain_age_model_with_healthy_controls(residualize=True, plot=True, model="LR", label="GR", only_four_regions=False,
                                                 onlyGMroi=True, age_bins_scaler=False):
    """
    Trains a brain-age prediction model on OpenBHB and evaluates on RLink (M0 and M3).

    Args:
        residualize (bool): whether to residualize ROI features on age and site.
        plot (bool): if True, generate plots.
        model (str): "LR" for Linear Regression or "EN" for Elastic Net.
        label (str): Lithium response class to filter ("GR", "NR", "PaR", "PaR/NR", "all").
        only_four_regions (bool): if True, do the prediction and evaluation on bilateral hippocampus and amygdala only.

    Returns:
        None. Prints metrics and optionally saves results.
    """

    assert label in ["GR","NR","PaR","PaR/NR","all"]
    assert model in ["LR", "EN", "MLP", "svm"], f"Wrong model name: it should be either 'LR' or 'EN', not '{model}'"
    
    list_roi_openbhb, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    if only_four_regions : 
        list_roi_openbhb, list_roi_rlink = FOUR_REGIONS_OF_INTEREST, FOUR_REGIONS_OF_INTEREST_LONG
    # for (i,j) in zip(list_roi_openbhb,list_roi_rlink):
    #     print(i,"   ",j)
    df_HC = pd.read_csv(DF_PATH_HEALTHY_BRAINS) #OPENBHB_DATAFRAME)
    # print(df_HC["site"].unique())

    print("dataframe Healthy Controls rois ...\n",df_HC)

    df_train, df_test, train_idx, test_idx, age_bins = stratified_split_balanced_age(df_HC, test_size=0.1,return_age_bins=True) #stratified_split(df_HC, include_site=True,min_site_size=80, test_size=0.1) #
    
    # plot distribution of healthy cohort used for training and testing of the brain age model 
    # df_plot = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    # plot_age_distribution(df_plot,"healthy controls")

    if onlyGMroi:
        list_roi_openbhb=[r for r in list_roi_openbhb if r.endswith("_GM_Vol")]
        list_roi_rlink=[r for r in list_roi_rlink if r.endswith("_GM_Vol")]
    X_arr = df_HC[list_roi_openbhb].values
    y_arr =  df_HC['age'].values 

    
    splits = [(np.asarray(train_idx), np.asarray(test_idx))]   # list of tuples
 
    if model =="LR":
        # age_regression_model = LinearRegression()
        age_regression_model = GridSearchCV(
            estimator=Ridge(), 
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]}, 
            cv=5, 
            n_jobs=1
            )

        
    if model=="EN":
        age_regression_model = GridSearchCV(
            estimator=ElasticNet(max_iter=10000),
            param_grid={
                "alpha": [0.01, 0.1, 1, 10],
                "l1_ratio": [0.1, 0.5, 0.9]  
            },
            cv=5,
            n_jobs=-1,
            scoring='r2',
            verbose=1
        )

    if model =="MLP":
        mlp_param_grid = {
            "hidden_layer_sizes": [
                # (100,), (50,), (25,), (10,), (5,),  
                # (100, 50), (50, 25), (25, 10), (10, 5),  
                (100, 50, 25), (50, 25, 10), (25, 10, 5) ,
                (100, 50, 25, 10), (50, 40, 30, 20), (25, 20, 15, 10),
                (100, 80, 60, 40, 20), (50, 40, 30, 20, 10), (25, 20, 15, 10, 5)
            ],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [1e-4, 1e-3],            
            "learning_rate_init": [0.0001, 0.001, 0.01]  

        }
        age_regression_model = GridSearchCV(
            #n_iter_no_change = 10 --> stop if there's no improvement after 10 epochs
            estimator=MLPRegressor(random_state=1, max_iter=1000, early_stopping=True, n_iter_no_change=10, validation_fraction=0.1),
            param_grid=mlp_param_grid,
            cv=3,
            n_jobs=-1, # use all cpu cores available for parallel computation
            scoring="r2"
        )

    if model =="svm":
        age_regression_model = GridSearchCV(
            estimator=svm.SVR(kernel='rbf'),
            param_grid={
                'gamma': ['scale'],
                'C': [0.1, 1.0, 10.0]
            },
            cv=5,
            n_jobs=-1,
            scoring='r2'  
        )

    if model=="xgboost":
        age_regression_model = GridSearchCV(
            estimator=XGBRegressor(random_state=42, n_jobs=1),
            param_grid= {
                    "n_estimators": [10, 30, 50],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 6],
                    "subsample": [0.8]
                },
            cv=5,
            n_jobs=1,
            scoring='r2' 
        )

    if age_bins_scaler: 
        pipeline = make_pipeline(
            age_regression_model)
    else: 
        pipeline = make_pipeline(
            StandardScaler(), 
            age_regression_model)
    # y_tr = shuffle(y_tr, random_state=42) # sanity check

    for fold_idx, (tr, te) in enumerate(splits):
        X_arr_tr, X_arr_te = X_arr[tr], X_arr[te]

        if residualize:
            # X_arr_tr, X_arr_te = residualize_data(df_HC, X_arr_tr, tr, X_arr_te, te) # OG results
            X_arr_tr, X_arr_te = residualize_data(df_HC, X_arr_tr, tr, X_arr_te, te, formula_res = "sex", formula_full="site + sex + age")
        
        if age_bins_scaler: X_arr_tr, X_arr_te, scalers = scale_x_y_agebins(X_arr_tr, X_arr_te, df_train, df_test, age_bins)
        
        pipeline.fit(X_arr_tr, y_arr[tr])
        print("Best params:", age_regression_model.best_params_)
        y_pred = pipeline.predict(X_arr_te)

        print("R² for OpenBHB test set:", round(r2_score(y_arr[te], y_pred),2), \
        "(MSE =", round(mean_squared_error(y_arr[te], y_pred),2),"), (MAE =", round(mean_absolute_error(y_arr[te],y_pred),2),")")
    
    df_M0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    df_M3M0 = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    df_M3 = df_M3M0[df_M3M0["session"]=="M03"]
    # plot_age_distribution(df_M0, "rlink")

    if label!="all":
        if label!="PaR/NR": 
            df_M0 = df_M0[(df_M0["response"] == label)].copy().reset_index(drop=True)
            df_M3 = df_M3[(df_M3["response"] == label)].copy()
        else : 
            df_M0 = df_M0[(df_M0["response"] == "PaR") | (df_M0["response"] == "NR")].copy().reset_index(drop=True)
            df_M3 = df_M3[(df_M3["response"] == "PaR") | (df_M3["response"] == "NR")].copy().reset_index(drop=True)
            

    X_rlink_M3 = df_M3[list_roi_rlink].values
    X_rlink_M0 = df_M0[list_roi_rlink].values
    y_M3 = df_M3["age"].values
    y_M0 = df_M0["age"].values

    if residualize:
        X_rlink_M3 = residualize_data(df_M3, X_rlink_M3,formula_res = "sex", formula_full="site + sex + age") #, formula_full = "site + sex + age + y")
        X_rlink_M0 = residualize_data(df_M0, X_rlink_M0, formula_res = "sex", formula_full="site + sex + age") #, formula_full = "site + sex + age + y")

    if age_bins_scaler: 
        X_rlink_M3 = scale_x_y_agebins_rlink(X_rlink_M3, df_M3, age_bins, scalers)
        X_rlink_M0 = scale_x_y_agebins_rlink(X_rlink_M0, df_M0, age_bins, scalers)

    y_pred_M3 = pipeline.predict(X_rlink_M3)
    y_pred_M0 = pipeline.predict(X_rlink_M0)

    if label =="GR": print("\n\n metrics for GR, Good Responders: ")
    if label =="NR": print("\n\n metrics for NR, Non Responders: ")
    if label =="PaR": print("\n\n metrics for PaR, Partial Responders: ")
    if label =="all": print("\n\n metrics for all Responders: ")

    print("R² for Rlink M3 :", round(r2_score(y_M3, y_pred_M3),2), \
          "(MSE =", round(mean_squared_error(y_M3, y_pred_M3),2),"), (MAE =", round(mean_absolute_error(y_M3,y_pred_M3),2),")")
    print("R² for Rlink M0 :", round(r2_score(y_M0, y_pred_M0),2), \
          "(MSE =", round(mean_squared_error(y_M0, y_pred_M0),2),"), (MAE =", round(mean_absolute_error(y_M0,y_pred_M0),2),")")

    df_M3_results , df_M0_results = pd.DataFrame(), pd.DataFrame()
    df_M3_results["participant_id"]=df_M3["participant_id"]
    df_M0_results["participant_id"]=df_M0["participant_id"]

    df_M3_results["predicted_age"]=y_pred_M3
    df_M0_results["predicted_age"]=y_pred_M0

    df_M3_results[["age","response"]]=df_M3[["age","response"]]
    df_M0_results[["age","response"]]=df_M0[["age","response"]]

    df_M3_results["session"]="M3"
    df_M0_results["session"]="M0"

    df_M3_results['age_diff'] = df_M3_results['predicted_age'] - df_M3_results['age']
    df_M0_results['age_diff'] = df_M0_results['predicted_age'] - df_M0_results['age']
    # positive age diff: quantifies
    # how much worse the brain age of BD patients is compared to healthy predictions
    # more likely that the age diff is positive in our case since BD subjects 
    # are expected to have higher brain age than HC
    # print(df_M3_results)
    # print(df_M0_results)

    df_M3_results["response"] = df_M3_results["response"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_M0_results["response"] = df_M0_results["response"].replace({"GR": 1, "PaR": 0, "NR": 0})

    print("\n\nage difference for ",label)

    if label=="all":
        print("M3 : mean ",df_M3_results["age_diff"].mean())
        print("M3 : median ",df_M3_results["age_diff"].median())

        print("\nM0 : mean ",df_M0_results["age_diff"].mean())
        print("M0 : median ",df_M0_results["age_diff"].median())
    else:
        if label =="GR":
            print("M3 : mean ",df_M3_results[df_M3_results["response"]==1]["age_diff"].mean())
            print("M3 : median ",df_M3_results[df_M3_results["response"]==1]["age_diff"].median())

            print("\nM0 : mean ",df_M0_results[df_M0_results["response"]==1]["age_diff"].mean())
            print("M0 : median ",df_M0_results[df_M0_results["response"]==1]["age_diff"].median())

        else: 
            print("M3 : mean ",df_M3_results[df_M3_results["response"]==0]["age_diff"].mean())
            print("M3 : median ",df_M3_results[df_M3_results["response"]==0]["age_diff"].median())

            print("\nM0 : mean ",df_M0_results[df_M0_results["response"]==0]["age_diff"].mean())
            print("M0 : median ",df_M0_results[df_M0_results["response"]==0]["age_diff"].median())

    if plot:
        df_M0_results = df_M0_results.copy()
        df_M3_results = df_M3_results.copy()
        df_M0_results['response_group'] = df_M0_results['response'].apply(group_response)
        df_M3_results['response_group'] = df_M3_results['response'].apply(group_response)

        df_M0_results['timepoint'] = 'M0'
        df_M3_results['timepoint'] = 'M3'

        df_all = pd.concat([df_M0_results, df_M3_results], ignore_index=True)

        # plot differences predicted age - real age for M0 and M3 separately 
        # plt.figure(figsize=(10,6))
        # sns.violinplot(
        #     data=df_all,
        #     x='response_group',
        #     y='age_diff',
        #     hue='timepoint',
        #     split=True,          
        #     inner='quartile',
        #     palette='Set2'
        # )
        # plt.xlabel('Response', fontsize=18)
        # plt.ylabel('Brain Age Difference', fontsize=18)
        # plt.title('Distribution of Brain age difference by Response and Timepoint', fontsize=20)
        # plt.xticks(rotation=0)
        # plt.tight_layout()
        # plt.show()

        # sns.violinplot(data=df_M0_results, x='y', y='age_diff')
        # plt.xlabel('Reponse', fontsize=18)
        # plt.ylabel('Brain age Difference', fontsize=18)
        # plt.title('Distribution of brain age by response to lithium', fontsize=20)  
        # plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        # plt.tight_layout()
        # plt.show()
        plt.figure(figsize=(10,6))
        sns.scatterplot(
            data=df_all,
            x='age',
            y='age_diff',
            hue='response_group',
            style='timepoint',
            palette='Set2',
            alpha=0.7
        )
        sns.regplot(
            data=df_all,
            x='age',
            y='age_diff',
            scatter=False,
            color='black',
            lowess=True  # smooth local regression line
        )
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # horizontal line at y=0
        plt.xlabel('Chronological Age', fontsize=18)
        plt.ylabel('Brain Age Difference (Predicted - Real)', fontsize=18)
        plt.title(f'Brain Age Difference vs Chronological Age using {model}', fontsize=20)
        plt.tight_layout()
        plt.show()

def main():
    # classify_site_rlink()
    # quit()

    # Lundi --> try with only GM values
    
    train_brain_age_model_with_healthy_controls(residualize=True, model="MLP", label="all", plot=True, onlyGMroi=True, age_bins_scaler=True) #, only_four_regions=True)

    # with Linear Regression + site and sex residualization:
    # -------------------------------------
    # R² for OpenBHB : 0.85 (MSE = 50.76)
    # -------------------------------------
    # metrics for RLink GOOD RESPONDERS : 
    # R² for Rlink M3 : 0.49 (MSE = 65.24 ) (MAE = 6.93 )
    # R² for Rlink M0 : 0.19 (MSE = 104.91 ) (MAE = 6.72 )
    # -------------------------------------
    # decrease in age prediction error from M0 to M3 suggests a "normalization" of brain anatomy amongst GR
    # -------------------------------------
    # age difference for GOOD RESPONDERS : 
    # M3 : mean  -6.07
    # M0 : mean  1.85
    # -------------------------------------
    # metrics for RLink NR/PaR:
    # R² for Rlink M3 : -0.39 (MSE = 281.61 ) 
    # R² for Rlink M0 : -0.15 (MSE = 235.01 )
    # -------------------------------------
    # age difference for PARTIAL AND NON-RESPONDERS TOGETHER : 
    # M3 : mean  -14.46
    # M0 : mean  -9.84
    # -------------------------------------
    # metrics for RLink PaR only :
    # R² for Rlink M3 : -0.17 (MSE = 248.78 ) 
    # R² for Rlink M0 : -0.29 (MSE = 275.11 )
    # -------------------------------------
    # age difference for  PARTIAL RESPONDERS
    # M3 : mean  -13.39
    # M0 : mean  -13.88
    # -------------------------------------
    # metrics for RLink NR only :
    # R² for Rlink M3 : 0.25 (MSE = 130.13 )
    # R² for Rlink M0 : -0.01 (MSE = 171.75 )
    # -------------------------------------
    # age difference for NON RESPONDERS
    # M3 : mean  -10.16
    # M0 : mean  -11.82
    # -------------------------------------

    # overall age difference (ALL RLink subjects) (OpenBHB + sex + site residualization)
    #  metrics for ALL Responders: 
    # R² for Rlink M3 : -0.59 (MSE = 283.4 ), (MAE = 14.52 )
    # R² for Rlink M0 : -0.06 (MSE = 188.81 ), (MAE = 11.33 )
    # age difference for  ALL
    # M3 : mean  -14.451046424200138
    # M0 : mean  -9.753049776454887

    # smallest age prediction error is found with M3 GR subjects 
    # most subjects seem to be predicted to have younger brain ages than the norm 
    # (so there is an overfitting or site issue with OpenBHB during training)
    # but we see that at least only the sub-group of RLink subjects predicted to have older brain ages than the norm
    # are good responders BEFORE Li intake, not after

    # ---------------------------------------------------------------------------------

    # with Elastic Net + site and sex residualization:
    # train_brain_age_model_with_openBHB(residualize=True, model="EN")
    # -------------------------------------
    # R² for OpenBHB : 0.85 (MSE = 50.2)

    # ----------------------------------------------------------------------------------
    # with svm + site and sex residualization
    # train_brain_age_model_with_openBHB(residualize=True, model="svm")
    # R² for OpenBHB test set: 0.87 (MSE = 42.37 )

    # ----------------------------------------------------------------------------------
    # with MLP + site and sex residualization --> training long pour des résultats pas meilleurs
    # au niveau de la différence entre âge prédit et vrai âge 
    # train_brain_age_model_with_openBHB(residualize=True, model="MLP")
    # R² for OpenBHB test set: 0.86 (MSE = 48.49 )

    # with residualize = False it's even worse (doesn't work either with residualization only on sex)


    # =================== with large pretraining HC dataset =============================
    # R² for big HC df test set: 0.84 (MSE = 38.29 ), (MAE = 4.74 )
    #  metrics for all Responders: 
    # R² for Rlink M3 : 0.48 (MSE = 93.09 ), (MAE = 7.86 )
    # R² for Rlink M0 : 0.46 (MSE = 95.77 ), (MAE = 7.89 )
    # age difference for  all
    # M3 : mean  -6.494948501373102
    # M0 : mean  -6.409452170111074

    ### on 4 rois
    # R² for OpenBHB test set: 0.18 (MSE = 193.33 ), (MAE = 11.19 )
    #  metrics for all Responders: 
    # R² for Rlink M3 : 0.18 (MSE = 146.81 ), (MAE = 10.33 )
    # R² for Rlink M0 : 0.13 (MSE = 153.85 ), (MAE = 10.67 )
    # age difference for  all
    # M3 : mean  1.9035492589860363
    # M0 : mean  -0.9678410937544496

    ### ===== on GM rois only
    # L2LR
    # R² for OpenBHB test set: 0.78 (MSE = 52.7 ), (MAE = 5.42 )
    #  metrics for all Responders: 
    # R² for Rlink M3 : 0.66 (MSE = 61.4 ), (MAE = 6.27 )
    # R² for Rlink M0 : 0.63 (MSE = 65.62 ), (MAE = 6.48 )
    # age difference for  all
    # M3 : mean  -2.087813926416575
    # M0 : mean  -3.253807432097087

    # MLP
    # R² for OpenBHB test set: 0.85 (MSE = 34.47 ), (MAE = 4.49 )
    #  metrics for all Responders:
    # R² for Rlink M3 : 0.46 (MSE = 95.64 ), (MAE = 7.46 )
    # R² for Rlink M0 : 0.6 (MSE = 70.76 ), (MAE = 6.49 )
    # age difference for  all
    # M3 : mean  4.734574188454907
    # M0 : mean  2.576647334829174




if __name__ == "__main__":
    main()
