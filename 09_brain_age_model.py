import pandas as pd
import numpy as np
import sys, pathlib
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from utils import stratified_split, get_lists_roi_in_both_openBHB_and_rlink
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as lm
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



def train_brain_age_model_with_openBHB(residualize=True, plot=True, model="LR", label="GR", only_four_regions=False):
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
    df_openbhb = pd.read_csv(OPENBHB_DATAFRAME)
    # print("dataframe OpenBHB rois ...\n",df_openbhb)

    df_train, df_test, train_idx, test_idx = stratified_split(df_openbhb)
    X_arr = df_openbhb[list_roi_openbhb].values
    y_arr =  df_openbhb['age'].values 

    splits = [(np.asarray(train_idx), np.asarray(test_idx))]   # list of tuples
 
    if model =="LR":
        age_regression_model = LinearRegression()
        
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
                (100,), (50,), (25,), (10,), (5,),  
                (100, 50), (50, 25), (25, 10), (10, 5),  
                (100, 50, 25), (50, 25, 10), (25, 10, 5)  
            ],
            "activation": ["relu"],
            "solver": ["sgd"],
            "alpha": [0.0001]
        }

        age_regression_model = GridSearchCV(
            estimator=MLPRegressor(random_state=1, max_iter=1000),
            param_grid=mlp_param_grid,
            cv=3,
            n_jobs=1,
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
            n_jobs=1,
            scoring='r2'  
        )

    

    pipeline = make_pipeline(StandardScaler(), age_regression_model)
    # y_tr = shuffle(y_tr, random_state=42) # sanity check

    for fold_idx, (tr, te) in enumerate(splits):
        X_arr_tr, X_arr_te = X_arr[tr], X_arr[te]

        if residualize:
            X_arr_tr, X_arr_te = residualize_data(df_openbhb, X_arr_tr, tr, X_arr_te, te)
            
        pipeline.fit(X_arr_tr, y_arr[tr])
        y_pred = pipeline.predict(X_arr_te)

        print("R² for OpenBHB test set:", round(r2_score(y_arr[te], y_pred),2), \
        "(MSE =", round(mean_squared_error(y_arr[te], y_pred),2),")")
    
    df_M0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    df_M3M0 = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    df_M3 = df_M3M0[df_M3M0["session"]=="M03"]

    if label!="all":
        if label!="PaR/NR": 
            df_M0 = df_M0[(df_M0["y"] == label)].copy().reset_index(drop=True)
            df_M3 = df_M3[(df_M3["y"] == label)].copy()
        else : 
            df_M0 = df_M0[(df_M0["y"] == "PaR") | (df_M0["y"] == "NR")].copy().reset_index(drop=True)
            df_M3 = df_M3[(df_M3["y"] == "PaR") | (df_M3["y"] == "NR")].copy().reset_index(drop=True)
            

    X_rlink_M3 = df_M3[list_roi_rlink].values
    X_rlink_M0 = df_M0[list_roi_rlink].values
    y_M3 = df_M3["age"].values
    y_M0 = df_M0["age"].values

    if residualize:
        X_rlink_M3 = residualize_data(df_M3, X_rlink_M3) #, formula_full = "site + sex + age + y")
        X_rlink_M0 = residualize_data(df_M0, X_rlink_M0) #, formula_full = "site + sex + age + y")

    y_pred_M3 = pipeline.predict(X_rlink_M3)
    y_pred_M0 = pipeline.predict(X_rlink_M0)

    if label =="GR": print("\n\n metrics for GR, Good Responders: ")
    if label =="NR": print("\n\n metrics for NR, Non Responders: ")
    if label =="PaR": print("\n\n metrics for PaR, Partial Responders: ")
    if label =="all": print("\n\n metrics for all Responders: ")

    print("R² for Rlink M3 :", round(r2_score(y_M3, y_pred_M3),2), \
          "(MSE =", round(mean_squared_error(y_M3, y_pred_M3),2),")")
    print("R² for Rlink M0 :", round(r2_score(y_M0, y_pred_M0),2), \
          "(MSE =", round(mean_squared_error(y_M0, y_pred_M0),2),")")

    df_M3_results , df_M0_results = pd.DataFrame(), pd.DataFrame()
    df_M3_results["participant_id"]=df_M3["participant_id"]
    df_M0_results["participant_id"]=df_M0["participant_id"]

    df_M3_results["predicted_age"]=y_pred_M3
    df_M0_results["predicted_age"]=y_pred_M0

    df_M3_results[["age","y"]]=df_M3[["age","y"]]
    df_M0_results[["age","y"]]=df_M0[["age","y"]]

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

    df_M3_results["y"] = df_M3_results["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_M0_results["y"] = df_M0_results["y"].replace({"GR": 1, "PaR": 0, "NR": 0})

    print("\n\nage difference for ",label)

    if label=="all":
        print("M3 : mean ",df_M3_results["age_diff"].mean())
        print("M3 : median ",df_M3_results["age_diff"].median())

        print("\nM0 : mean ",df_M0_results["age_diff"].mean())
        print("M0 : median ",df_M0_results["age_diff"].median())
    else:
        if label =="GR":
            print("M3 : mean ",df_M3_results[df_M3_results["y"]==1]["age_diff"].mean())
            print("M3 : median ",df_M3_results[df_M3_results["y"]==1]["age_diff"].median())

            print("\nM0 : mean ",df_M0_results[df_M0_results["y"]==1]["age_diff"].mean())
            print("M0 : median ",df_M0_results[df_M0_results["y"]==1]["age_diff"].median())

        else: 
            print("M3 : mean ",df_M3_results[df_M3_results["y"]==0]["age_diff"].mean())
            print("M3 : median ",df_M3_results[df_M3_results["y"]==0]["age_diff"].median())

            print("\nM0 : mean ",df_M0_results[df_M0_results["y"]==0]["age_diff"].mean())
            print("M0 : median ",df_M0_results[df_M0_results["y"]==0]["age_diff"].median())

    # if plot:
    #     df_M0_results = df_M0_results.copy()
    #     df_M3_results = df_M3_results.copy()
    #     df_M0_results['response_group'] = df_M0_results['y'].apply(group_response)
    #     df_M3_results['response_group'] = df_M3_results['y'].apply(group_response)

    #     df_M0_results['timepoint'] = 'M0'
    #     df_M3_results['timepoint'] = 'M3'

    #     df_all = pd.concat([df_M0_results, df_M3_results], ignore_index=True)
    #     plt.figure(figsize=(10,6))
    #     sns.violinplot(
    #         data=df_all,
    #         x='response_group',
    #         y='age_diff',
    #         hue='timepoint',
    #         split=True,          
    #         inner='quartile',
    #         palette='Set2'
    #     )
    #     plt.xlabel('Response', fontsize=18)
    #     plt.ylabel('Brain Age Difference', fontsize=18)
    #     plt.title('Distribution of Brain age difference by Response and Timepoint', fontsize=20)
    #     plt.xticks(rotation=0)
    #     plt.tight_layout()
    #     plt.show()

        # sns.violinplot(data=df_M0_results, x='y', y='age_diff')
        # plt.xlabel('Reponse', fontsize=18)
        # plt.ylabel('Brain age Difference', fontsize=18)
        # plt.title('Distribution of brain age by response to lithium', fontsize=20)  
        # plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        # plt.tight_layout()
        # plt.show()

def main():
    
    train_brain_age_model_with_openBHB(residualize=True, model="LR", label="GR")

    # with Linear Regression + site and sex residualization:
    # -------------------------------------
    # R² for OpenBHB : 0.85 (MSE = 50.76)
    # -------------------------------------
    # metrics for RLink GOOD RESPONDERS : 
    # R² for Rlink M3 : 0.49 (MSE = 65.24 )
    # R² for Rlink M0 : 0.19 (MSE = 104.91 )
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


    # overall age difference (all RLink subjects)


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




if __name__ == "__main__":
    main()
