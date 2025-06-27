from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    

class PairwiseWhiteningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pair_dict, method="zca", whiten=True):
        """
        Parameters:
        - pair_dict: dict {region_name: (gm_index, csf_index)} specifying pairs of features to whiten together
        - method: "zca" or "full" whitening method
        - whiten: whether to apply whitening 
        """
        self.pair_dict = pair_dict
        self.method = method
        self.whiten = whiten
        self.transforms_ = {} # will hold whitening matrix and mean per pair

    def fit(self, X, y=None):
        if not self.whiten:
            # Identity whitening matrix and zero mean: no whitening applied
            self.whitening_matrix_ = np.eye(X.shape[1])
            self.means_ = np.zeros(X.shape[1])
            return self

        n_features = X.shape[1]
        W_full = np.eye(n_features)  # full whitening matrix (initially identity)
        means_full = np.zeros(n_features) # means vector for centering

        # Iterate over each pair of features for local whitening
        for region, (i, j) in self.pair_dict.items():
            X_pair = X[:, [i, j]] # extract data for the pair
            mean_pair = X_pair.mean(axis=0) # empirical mean vector of the pair
            X_pair_centered = X_pair - mean_pair # center pair

            cov = np.cov(X_pair_centered, rowvar=False) # 2x2 covariance matrix of the pair
            

            epsilon = 1e-5  # stability term to avoid dividing by zero eigenvalues

            if self.method == "zca":
                # ZCA whitening:
                # Eigendecompose covariance:
                # cov = U diag(S) U^T
                U, S, _ = np.linalg.svd(cov)
                # Whitening matrix: W = U diag(1/sqrt(S + epsilon)) U^T
                W = U @ np.diag(1. / np.sqrt(S + epsilon)) @ U.T
                det = np.linalg.det(W)
                # print("condition :",np.linalg.cond(W) )
                # print("Determinant:", det)
                # if det ==0: 
                #     print(det)


            elif self.method == "full":
                # Full whitening via eigendecomposition:
                eigvals, eigvecs = np.linalg.eigh(cov)
                W = eigvecs @ np.diag(1. / np.sqrt(eigvals + epsilon)) @ eigvecs.T
            else:
                raise ValueError("Unknown whitening method")

            # Store whitening matrix and mean for this pair for later transform calls
            self.transforms_[region] = (W, mean_pair)

            # Embed the 2x2 whitening matrix into the full whitening matrix W_full
            # This places W on the 2x2 block corresponding to indices (i,j)
            W_full[np.ix_([i,j],[i,j])] = W
            means_full[[i,j]] = mean_pair

        # Store full whitening matrix (block diagonal with 2x2 blocks) and means vector
        self.whitening_matrix_ = W_full
        self.means_ = means_full
        return self

    def transform(self, X):
        if not self.whiten:
            return X
        X_whitened = X.copy()

        # Apply whitening pairwise for each pair:
        # For each pair (features i,j), center by mean and multiply by W^T
        for region, (i, j) in self.pair_dict.items():
            W, mean = self.transforms_[region]
            X_pair = X[:, [i, j]]
            X_pair_whitened = (X_pair - mean) @ W.T # apply whitening transform
            X_whitened[:, [i, j]] = X_pair_whitened
        return X_whitened

    def inverse_transform_coefficients(self, coeffs):
        """
        Invert coefficients (weights) from whitened space to original feature space.
        """

        if not self.whiten:
            return coeffs
        w_orig = coeffs.copy()

        # For each pair, multiply the corresponding coefficients by W         
        for region, (i, j) in self.pair_dict.items():
            W, mean = self.transforms_[region]
            w_g = w_orig[[i, j]]
            w_orig[[i, j]] = np.linalg.inv(W.T) @ w_g

        return w_orig
        
        # W_inv = np.linalg.inv(self.whitening_matrix_)
        # return W_inv.T @ coeffs

class PartialWhiteningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, groups, lambda_reg=0.1, method='full', whiten=True):
        """
        groups: dict {group_name: list of feature indices}
        lambda_reg: regularization for covariance (ridge)
        method: 'full' or 'zca'
        whiten: bool to apply whitening or not
        """
        self.groups = {k: list(v) if isinstance(v, tuple) else v for k, v in groups.items()}
        self.lambda_reg = lambda_reg
        self.method = method
        self.whiten = whiten
        self.transforms_ = {}

    def fit(self, X, y=None):
        if not self.whiten:
            return self
        self.transforms_ = {}
        epsilon = 1e-5 
        for gname, indices in self.groups.items():
            Xg = X[:, indices]
            mean = Xg.mean(axis=0)
            Xg_centered = Xg - mean
            cov = np.cov(Xg_centered, rowvar=False)

            cov_reg = cov + self.lambda_reg * np.eye(len(indices))
            if self.method == 'zca':
                U, S, _ = np.linalg.svd(cov_reg)
                W = U @ np.diag(1. / np.sqrt(S + epsilon)) @ U.T
                # det = np.linalg.det(W)
                # print("condition :",np.linalg.cond(W) )
                # if det ==0: 
                #     print(det)
                #     quit()
            elif self.method == 'full':
                eigvals, eigvecs = np.linalg.eigh(cov_reg)
                W = eigvecs @ np.diag(1. / np.sqrt(eigvals + epsilon)) @ eigvecs.T
            else:
                raise ValueError("Unknown method")
            self.transforms_[gname] = (W, mean, indices)
        return self

    def transform(self, X):
        if not self.whiten:
            return X
        X_whitened = X.copy()
        for gname, (W, mean, indices) in self.transforms_.items():
            Xg = X[:, indices]
            Xg_whitened = (Xg - mean) @ W.T
            X_whitened[:, indices] = Xg_whitened
        return X_whitened

    def inverse_transform_weights(self, w):
        # w: weights in whitened space (shape [n_features])
        w_orig = w.copy()
        for gname, (W, mean, indices) in reversed(self.transforms_.items()):
            # invert whitening on group weights
            w_g = w_orig[indices]
            w_orig[indices] = np.linalg.inv(W.T) @ w_g 
            
        return w_orig

def main():

    def zca_whiten(X):
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        cov = np.cov(X_centered, rowvar=False)
        U, S, _ = np.linalg.svd(cov)
        W = U @ np.diag(1 / np.sqrt(S + 1e-5)) @ U.T
        X_whitened = X_centered @ W.T
        return X_whitened, W, X_mean    

    # compare classification performance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    def run_test(X, y, name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        # Whitening
        X_train_white, W, X_mean = zca_whiten(X_train)
        X_test_white = (X_test - X_mean) @ W.T

        X_reconstructed = X_train_white @ np.linalg.inv(W.T) + X_mean
        print("Max reconstruction error:", np.abs(X_train - X_reconstructed).max())
        # if the reconstruction error is near zero, there is no information loss

        clf = LogisticRegression()

        # On original data
        clf.fit(X_train, y_train)
        y_pred_proba_orig = clf.predict_proba(X_test)[:, 1]
        acc_orig = accuracy_score(y_test, clf.predict(X_test))
        auc_orig = roc_auc_score(y_test, y_pred_proba_orig)

        # On whitened data
        clf.fit(X_train_white, y_train)
        y_pred_proba_white = clf.predict_proba(X_test_white)[:, 1]
        acc_white = accuracy_score(y_test, clf.predict(X_test_white))
        auc_white = roc_auc_score(y_test, y_pred_proba_white)

        print(f"\n== {name} ==")
        print("Original Accuracy:", acc_orig)
        print("Original ROC AUC:", auc_orig)
        print("Whitened Accuracy:", acc_white)
        print("Whitened ROC AUC:", auc_white)

    # Test 1: y signal + latent L shared variance : the features' information on the label is shared btw them
    np.random.seed(0)
    n_samples = 1000
    y = np.random.randint(0, 2, n_samples)
    epsilon1 = np.random.normal(0, 1, n_samples)
    epsilon2 = np.random.normal(0, 1, n_samples)
    L = 2 * y + epsilon1

    x1 = L + epsilon1
    x2 = L + epsilon2
    X = np.vstack([x1, x2]).T
    run_test(X, y, "Test 1: L = 2y + noise (shared informative signal)")

    # Test 2: L = noise, both features get L + y + noise : the features' information on the label is not shared btw them
    epsilon1 = np.random.normal(0, 1, n_samples)
    epsilon2 = np.random.normal(0, 1, n_samples)
    epsilon3 = np.random.normal(0, 1, n_samples)
    L = epsilon3  # non-informative shared variance

    x1 = L + y + epsilon1
    x2 = L + y + epsilon2
    X2 = np.vstack([x1, x2]).T
    run_test(X2, y, "Test 2: L = noise only (shared nuisance)")





if __name__ == "__main__":
    main()

