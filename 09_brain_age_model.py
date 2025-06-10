import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from utils import stratified_split, get_lists_roi_in_both_openBHB_and_rlink
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as lm

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

def group_response(label):
    if label == "GR":
        return "GR"
    elif label in ["PaR", "NR"]:
        return "PaR/NR"
    else:
        return label 
    
def train_brain_age_model_with_openBHB(roi=None, residualize=False, plot=True):
    """
    roi : either a single roi name or a list of roi to do the regression with
    if roi is None, default is using all available rois
    
    """
    list_roi_openbhb, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    df_openbhb = pd.read_csv(OPENBHB_DATAFRAME)
    # print("dataframe OpenBHB rois ...\n",df_openbhb)
    if roi is None: roi = list_roi_openbhb

    df_train, df_test, train_idx, test_idx = stratified_split(df_openbhb)
    X_tr = df_train[roi].values
    X_te = df_test[roi].values 
    y_tr =  df_train['age'].values 
    y_te =  df_test['age'].values

    if residualize:  
        print("Residualizing on sex and site")
        residualizer = Residualizer(
            data=df_openbhb,
            formula_res="sex",
            formula_full= "sex + age"
        )
        Zres = residualizer.get_design_mat(df_openbhb)
        Zres_train, Zres_test = Zres[train_idx], Zres[test_idx]
        residualizer.fit(X_tr, Zres_train)
        X_tr = residualizer.transform(X_tr, Zres_train)
        X_te = residualizer.transform(X_te, Zres_test)
    
    print(np.shape(X_tr), type(X_tr))
    print(np.shape(y_tr), type(y_tr))
    if X_tr.ndim==1: X_tr = X_tr.reshape(-1, 1)
    if X_te.ndim==1: X_te = X_te.reshape(-1, 1)

    print(np.shape(X_tr), type(X_tr))
    # pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            estimator=ElasticNet(max_iter=10000),
            param_grid={
                "alpha": [0.01, 0.1, 1, 10],
                "l1_ratio": [0.1, 0.5, 0.9]  # 0 = Ridge, 1 = Lasso
            },
            cv=5,
            n_jobs=-1,
            scoring='r2',
            verbose=1
        )
    )
    # y_tr = shuffle(y_tr, random_state=42) # sanity check

    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_te)
    print("MSE:", mean_squared_error(y_te, y_pred))
    print("R²:", r2_score(y_te, y_pred))

    df = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    df_M3 = df[df["session"]=="M03"]
    df_M0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)

    # change code here if we do regression on only select roi
    X_rlink_M3 = df_M3[list_roi_rlink].values
    X_rlink_M0 = df_M0[list_roi_rlink].values

    if residualize:
        Zres_M3 = df_M3[["sex","age"]].values
        Zres_M3 = np.hstack((np.ones((Zres_M3.shape[0], 1)) , Zres_M3)) # add intercept
        X_rlink_M3 = residualizer.transform(X_rlink_M3, Zres_M3)

        Zres_M0 = df_M0[["sex","age"]].values
        Zres_M0 = np.hstack((np.ones((Zres_M0.shape[0], 1)) , Zres_M0))
        X_rlink_M0 = residualizer.transform(X_rlink_M0, Zres_M0)
    
    y_M3 = df_M3["age"].values
    y_M0 = df_M0["age"].values
    y_pred_M3 = pipeline.predict(X_rlink_M3)
    y_pred_M0 = pipeline.predict(X_rlink_M0)

    df_M3_results , df_M0_results = pd.DataFrame(), pd.DataFrame()
    df_M3_results["participant_id"]=df_M3["participant_id"]
    df_M0_results["participant_id"]=df_M0["participant_id"]

    df_M3_results["predicted_age"]=y_pred_M3
    df_M0_results["predicted_age"]=y_pred_M0

    df_M3_results[["age","y"]]=df_M3[["age","y"]]
    df_M0_results[["age","y"]]=df_M0[["age","y"]]

    df_M3_results['age_diff'] = df_M3_results['predicted_age'] - df_M3_results['age']
    df_M0_results['age_diff'] = df_M0_results['predicted_age'] - df_M0_results['age']

    print(df_M3_results)
    print(df_M0_results)
    print(df_M3_results[df_M3_results["y"]=="PaR"]["age_diff"].mean())
    print(df_M0_results[df_M0_results["y"]=="PaR"]["age_diff"].mean())

    print("at M3: \nMSE:", mean_squared_error(y_M3, y_pred_M3))
    print("R²:", r2_score(y_M3, y_pred_M3))
    print("at M0 :\nMSE:", mean_squared_error(y_M0, y_pred_M0))
    print("R²:", r2_score(y_M0, y_pred_M0))

    if plot:
        df_M0_results = df_M0_results.copy()
        df_M3_results = df_M3_results.copy()
        df_M0_results['response_group'] = df_M0_results['y'].apply(group_response)
        df_M3_results['response_group'] = df_M3_results['y'].apply(group_response)

        df_M0_results['timepoint'] = 'M0'
        df_M3_results['timepoint'] = 'M3'

        df_all = pd.concat([df_M0_results, df_M3_results], ignore_index=True)
        plt.figure(figsize=(10,6))
        sns.violinplot(
            data=df_all,
            x='response_group',
            y='age_diff',
            hue='timepoint',
            split=True,          
            inner='quartile',
            palette='Set2'
        )
        plt.xlabel('Response', fontsize=18)
        plt.ylabel('Brain Age Difference', fontsize=18)
        plt.title('Distribution of Brain age difference by Response and Timepoint', fontsize=20)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # sns.violinplot(data=df_M0_results, x='y', y='age_diff')
        # plt.xlabel('Reponse', fontsize=18)
        # plt.ylabel('Brain age Difference', fontsize=18)
        # plt.title('Distribution of brain age by response to lithium', fontsize=20)  
        # plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        # plt.tight_layout()
        # plt.show()

def main():
    # roi=FOUR_REGIONS_OF_INTEREST[0]
    # print(roi)
    train_brain_age_model_with_openBHB(residualize=True)


if __name__ == "__main__":
    main()
