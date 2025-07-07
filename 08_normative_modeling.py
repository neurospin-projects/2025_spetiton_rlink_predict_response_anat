from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import sys

# inputs
ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
OPENBHB_DATAFRAME = DATA_DIR+"OpenBHB_roi.csv"
# RLink inputs
RLINK_DATAFRAME_M00_M03 = DATA_DIR + "df_ROI_age_sex_site_M00_M03_v4labels.csv"
RLINK_DATAFRAME_ALL_M00 = DATA_DIR + "df_ROI_age_sex_site_M00_v4labels.csv"
SELECTED_REGIONS_OF_INTEREST_RLINK = ['Left Hippocampus_GM_Vol','Right Hippocampus_GM_Vol','Left Amygdala_GM_Vol', 'Right Amygdala_GM_Vol']


# Warping utilities
def warp_sinh_arcsinh(y, epsilon=0.0, b=1.0):
    return np.sinh(b * np.arcsinh(y) + epsilon * b)

def inv_warp_sinh_arcsinh(gy, epsilon=0.0, b=1.0):
    return np.sinh((np.arcsinh(gy) - epsilon * b) / b)

def log_likelihood_sinh_arcsinh(params, y):
    epsilon, b = params
    if b <= 0:
        return np.inf
    g_y = warp_sinh_arcsinh(y, epsilon, b)
    logpdf = norm.logpdf(g_y)
    dg_dy = b * np.cosh(b * np.arcsinh(y) + epsilon * b) / np.sqrt(1 + y**2)
    return -np.sum(logpdf + np.log(dg_dy))

# Main Normative model class
class NormativeBLR(BaseEstimator, RegressorMixin):
    def __init__(self, warp=False, bsplines=True, residualize=False, n_knots=3):
        self.warp = warp
        self.bsplines = bsplines
        self.residualize = residualize  # Placeholder if needed
        self.n_knots = n_knots
        self.models_ = []
        self.stds_ = []
        self.epsilons_ = []
        self.betas_ = []

    def fit(self, X_covariates, Y_roi):
        """
        Fit one Bayesian Ridge model per ROI.
        Parameters:
        - X_covariates: shape (n_samples, 2) -> [age, sex]
        - Y_roi: shape (n_samples, n_rois)
        """
        n_rois = Y_roi.shape[1]
        self.models_ = []
        self.stds_ = []
        self.epsilons_ = []
        self.betas_ = []

        for i in range(n_rois):
            y = Y_roi[:, i]

            # Optional warping
            epsilon_opt, beta_opt = 0.0, 1.0
            if self.warp:
                initial_params = [0.0, 1.0]
                res = minimize(log_likelihood_sinh_arcsinh, initial_params, args=(y,),
                               method='L-BFGS-B', bounds=[(-2, 2), (1e-3, 10)])
                epsilon_opt, beta_opt = res.x
                y = warp_sinh_arcsinh(y, epsilon=epsilon_opt, b=beta_opt)

            self.epsilons_.append(epsilon_opt)
            self.betas_.append(beta_opt)

            # Design matrix
            if self.bsplines:
                preprocessor = ColumnTransformer([
                    ('spline_age', SplineTransformer(degree=3, n_knots=self.n_knots), [0]),
                    ('passthrough_sex', 'passthrough', [1]) # ignore age and sex columns, output without applying any transformation on these columns
                ])
            else:
                preprocessor = 'passthrough'

            model = make_pipeline(
                preprocessor,
                StandardScaler(),
                BayesianRidge()
            )

            model.fit(X_covariates, y)
            y_pred = model.predict(X_covariates)

            # Compute std of residuals in original space
            if self.warp:
                y_pred = inv_warp_sinh_arcsinh(y_pred, epsilon=epsilon_opt, b=beta_opt)
                y = inv_warp_sinh_arcsinh(y, epsilon=epsilon_opt, b=beta_opt)

            residuals = y - y_pred

            resid_std = np.std(residuals, ddof=1) # use ddof=1 for sample std

            self.models_.append(model)
            self.stds_.append(resid_std)

        return self

    def predict(self, X_covariates, Y_roi=None, return_zscores=False):
        """
        Predict expected ROI values and optionally z-scores.

        Parameters:
        - X_covariates: shape (n_samples, 2)
        - Y_roi: optional, shape (n_samples, n_rois)
        - return_zscores: if True, return z-scores (requires Y_roi)

        Returns:
        - predicted: shape (n_samples, n_rois)
        - optionally: z_scores: shape (n_samples, n_rois)
        """
        preds = []
        zscores = []
        r2s = []

        for i, model in enumerate(self.models_):
            y_pred = model.predict(X_covariates)

            if self.warp:
                y_pred = inv_warp_sinh_arcsinh(y_pred, epsilon=self.epsilons_[i], b=self.betas_[i])

            preds.append(y_pred)

            if return_zscores:
                if Y_roi is None:
                    raise ValueError("Y_roi must be provided to compute z-scores")
                residuals = Y_roi[:, i] - y_pred
                resid_std = residuals.std(ddof=1)  # use ddof=1 for sample std
                z = residuals / resid_std
                zscores.append(z)

        preds = np.vstack(preds).T  # shape (n_samples, n_rois)

        if return_zscores:
            zscores = np.vstack(zscores).T
            return preds, zscores

        return preds

def main():

    # - roi_values has shape (n_subjects, n_rois)
    # - covariates has shape (n_subjects, 2) for [age, sex]
    df_openbhb = pd.read_csv(OPENBHB_DATAFRAME)
    from utils import stratified_split, get_lists_roi_in_both_openBHB_and_rlink

    df_train, df_test, train_idx, test_idx  = stratified_split(df_openbhb, verbose=False)
    list_roi_openbhb, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    print("nb of openBHB rois :", len(list_roi_openbhb)," nb rois rlink ", len(list_roi_rlink))

    X_tr = df_train[['age', 'sex']]
    X_te = df_test[['age', 'sex']]
    y_tr =  df_train[list_roi_openbhb].values
    y_te =  df_test[list_roi_openbhb].values
    print("Xtr , ytr",np.shape(X_tr), np.shape(y_tr))
    print("X_te , y_te",np.shape(X_te), np.shape(y_te))

    model = NormativeBLR(warp=True, bsplines=True, residualize=False)
    model.fit(X_tr, y_tr)

    """
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
    
    """

    def get_M0_M3():
        """
            returns two dataframes of ROIs at baseline (M0) and 3 months after li intake (M3) 
        """
        df_M0M3 = pd.read_csv(RLINK_DATAFRAME_M00_M03)
        df_M0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
        df_M3 = df_M0M3[df_M0M3["session"] == "M03"].copy()
        df_M3 = df_M3.reset_index(drop=True)
        df_M0 = df_M0.reset_index(drop=True)
        df_M0["y"] = df_M0["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
        df_M3["y"] = df_M3["y"].replace({"GR": 1, "PaR": 0, "NR": 0})

        return df_M0, df_M3

    df_M0, df_M3 = get_M0_M3()

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.histplot(df_M0["age"], kde=True, bins=30)
    # plt.title("Age Distribution")
    # plt.xlabel("Age")
    # plt.ylabel("Count")
    # plt.show()

    # responses
    roi_values_M0 = df_M0[list_roi_rlink].values
    roi_values_M3 = df_M3[list_roi_rlink].values

    # covariates
    X_rlink_M0 = df_M0[['age', 'sex']]
    X_rlink_M3 = df_M3[['age', 'sex']]

    roi_pred_M0, z_scores_M0 = model.predict(X_rlink_M0, roi_values_M0, return_zscores=True)
    roi_pred_M3, z_scores_M3 = model.predict(X_rlink_M3, roi_values_M3, return_zscores=True)

    df_zscores_M0 = pd.DataFrame(z_scores_M0, columns = list_roi_rlink)
    df_zscores_M0.insert(0, 'participant_id', df_M0["participant_id"])  
    df_zscores_M0["y"]=df_M0["y"]
    df_zscores_M3 = pd.DataFrame(z_scores_M3, columns = list_roi_rlink)
    df_zscores_M3.insert(0, 'participant_id', df_M3["participant_id"])  
    df_zscores_M3["y"]=df_M3["y"]
    
    print("M3 GR zscores :\n", df_zscores_M3[df_zscores_M3["y"]==0][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())
    print("M3 NR/PaR zscores :\n",df_zscores_M3[df_zscores_M3["y"]==1][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())

    print("M0 GR zscores :\n", df_zscores_M0[df_zscores_M0["y"]==0][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())
    print("M0 NR/PaR zscores :\n",df_zscores_M0[df_zscores_M0["y"]==1][SELECTED_REGIONS_OF_INTEREST_RLINK].mean())

    print(df_zscores_M0)
    print("correlation matrix Frobenius norm M0 roi ",np.linalg.norm(df_M0[list_roi_rlink].corr().values, ord='fro'))
    print("correlation matrix Frobenius norm M0 zscores ",np.linalg.norm(df_zscores_M0[list_roi_rlink].corr().values, ord='fro'))

    sys.path.append('/neurospin/psy_sbox/temp_sara/')
    from pylearn_mulm.mulm.residualizer import Residualizer
    residualizer = Residualizer(data=df_M0[["age","sex","site","y"]], \
                                formula_res= "age + sex", formula_full= "age + sex + y")
    Zres = residualizer.get_design_mat(df_M0[["age","sex","site","y"]])
    residualizer.fit(df_M0[list_roi_rlink].values, Zres)
    roi_M0_res = residualizer.transform(df_M0[list_roi_rlink].values, Zres)
    df_roi_M0_res = pd.DataFrame(roi_M0_res, columns=list_roi_rlink)
    print("correlation matrix Frobenius norm residualized M0 roi "\
          ,np.linalg.norm(df_roi_M0_res.corr().values, ord='fro'))


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
        print(df0_common)
        print(df3_common)

        assert all(df0_common["participant_id"] == df3_common["participant_id"])
        assert all(df0_common["y"] == df3_common["y"])

        results = {}

        for label in [0, 1]:
            # Filter rows where y == label
            mask = df0_common["y"] == label
            d0 = df0_common.loc[mask, list_roi].abs()
            d3 = df3_common.loc[mask, list_roi].abs()

            # Compute per-ROI % of participants where |df3| < |df0|
            # --> equivalent to a closening to the "norm" after Li intake
            percentages = (d3 < d0).sum(axis=0) / len(d0) * 100
            results[label] = percentages

        return results  # dict: {0: Series, 1: Series}

    # results = get_roi_diff_percentages(df_zscores_M0, df_zscores_M3, list_roi_rlink)
    print(df_zscores_M0)
    print(df_zscores_M3)
    for roi in SELECTED_REGIONS_OF_INTEREST_RLINK:
        print(roi)
        results = get_roi_diff_percentages(df_zscores_M0, df_zscores_M3, [roi])
        print(results)

    # add violin plots here





if __name__ == "__main__":
    main()