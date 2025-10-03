from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np



# warping functions
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

class NormativeBLR(BaseEstimator, RegressorMixin):
    def __init__(self, warp=False, bsplines=True, n_knots=3):
        self.warp = warp
        self.bsplines = bsplines
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
                    ('onehot_sex', OneHotEncoder(drop='if_binary'), [1]) # treat sex as categorical variable
                    # ('passthrough_sex', 'passthrough', [1]) # ignore age and sex columns, output without applying any transformation on these columns
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
        - Y_roi: shape (n_samples, n_rois)
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
                z = residuals / self.stds_[i] 
                zscores.append(z)

        preds = np.vstack(preds).T  # shape (n_samples, n_rois)

        if return_zscores:
            zscores = np.vstack(zscores).T
            return preds, zscores

        return preds

def main():

    print("normative modeling")

    """
    use as : 
    
    model = NormativeBLR(warp=True, bsplines=True)
    model.fit(X_tr, y_tr) # X_tr : covariates (n_subjects, [age,sex]), y_tr : responses (n_subjects, n_rois)

    # Predict and get z-scores
    roi_pred, z_scores = model.predict(X_te, y_te, return_zscores=True)
    print("z_scores ",np.shape(z_scores), type(z_scores))

    """





if __name__ == "__main__":
    main()
