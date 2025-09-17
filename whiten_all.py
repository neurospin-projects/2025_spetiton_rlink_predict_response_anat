import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FullWhiteningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method="zca", lambda_reg=0.0, whiten=True):
        """
        Parameters:
        - method: "zca" or "full" whitening method
        - lambda_reg: regularization term added to covariance diagonal
        - whiten: whether to apply whitening
        """
        self.method = method
        self.lambda_reg = lambda_reg
        self.whiten = whiten

    def fit(self, X, y=None):
        if not self.whiten:
            self.mean_ = np.zeros(X.shape[1])
            self.whitening_matrix_ = np.eye(X.shape[1])
            return self
        
        # Compute mean and covariance
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered, rowvar=False)
        
        # Regularize covariance
        cov_reg = cov + self.lambda_reg * np.eye(cov.shape[0])
        
        epsilon = 1e-5
        
        if self.method == "zca":
            U, S, _ = np.linalg.svd(cov_reg)
            self.whitening_matrix_ = U @ np.diag(1. / np.sqrt(S + epsilon)) @ U.T
        
        elif self.method == "full":
            eigvals, eigvecs = np.linalg.eigh(cov_reg)
            self.whitening_matrix_ = eigvecs @ np.diag(1. / np.sqrt(eigvals + epsilon)) @ eigvecs.T
        
        else:
            raise ValueError("Unknown whitening method")
        
        return self

    def transform(self, X):
        if not self.whiten:
            return X
        X_centered = X - self.mean_
        return X_centered @ self.whitening_matrix_.T

    def inverse_transform_weights(self, w):
        """Convert weights from whitened space back to original space."""
        if not self.whiten:
            return w
        return np.linalg.inv(self.whitening_matrix_.T) @ w
