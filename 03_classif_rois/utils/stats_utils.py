from scipy.stats import mannwhitneyu, norm
import numpy as np


def mannwhitneyu_4auc(auc: float, n_pos: int, n_neg: int, alternative: str = "greater"):
    """Mann-Whitney U test for ROC-AUC.
    Test whether a ROC-AUC is significantly greater than 0.5 (chance)
    using the exact equivalence AUC = U / (n_pos * n_neg).

   The Mann-Whitney U statistic and ROC-AUC are directly related:
   **AUC = U / (n₁ × n₀)**, where n₁ and n₀ are the number of positives
   and negatives.
   So you can recover U exactly from the AUC and test it directly.

    Parameters
    ----------
    auc       : observed ROC-AUC
    n_pos     : number of positive cases (responders)
    n_neg     : number of negative cases (non-responders)
    alternative : 'greater' (AUC > 0.5), 'two-sided', or 'less'

    Returns
    -------
    U, z, p
    """
    U = auc * n_pos * n_neg

    # Normal approximation (valid when n_pos, n_neg > ~20)
    mu_U    = n_pos * n_neg / 2
    sigma_U = np.sqrt(n_pos * n_neg * (n_pos + n_neg + 1) / 12)
    z = (U - mu_U) / sigma_U

    if alternative == "greater":
        p = norm.sf(z)
    elif alternative == "less":
        p = norm.cdf(z)
    else:
        p = 2 * norm.sf(abs(z))

    return U, z, p


def mannwhitneyu_from_scores_(y_true, y_score, alternative="greater"):
    """Compute AUC and test significance directly from predicted scores.

    scipy's mannwhitneyu implements the exact same statistic as AUC,
    so no approximation is needed.

    Parameters
    ----------
    y_true      : array-like of 0/1 labels
    y_score     : array-like of predicted scores
    alternative : 'greater' (AUC > 0.5, default), 'two-sided', or 'less'

    Returns
    -------
    auc, U, p
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    U, p = mannwhitneyu(pos_scores, neg_scores, alternative=alternative)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    auc = U / (n_pos * n_neg)
    return auc, U, p


def test_mannwhitneyu_4auc():
    """Check that mannwhitneyu_4auc and mannwhitneyu_from_scores agree."""
    rng = np.random.default_rng(0)
    n_pos, n_neg = 45, 60
    pos_scores = rng.normal(0.6, 0.3, n_pos)
    neg_scores = rng.normal(0.4, 0.3, n_neg)

    y_true  = np.array([1] * n_pos + [0] * n_neg)
    y_score = np.concatenate([pos_scores, neg_scores])

    auc_exact, U_exact, p_exact = mannwhitneyu_from_scores_(y_true, y_score)
    _, z_approx, p_approx = mannwhitneyu_4auc(auc_exact, n_pos, n_neg)

    print(f"AUC             : {auc_exact:.4f}")
    print(f"U  (exact)      : {U_exact:.1f}")
    print(f"p  (exact MWU)  : {p_exact:.4f}")
    print(f"z  (approx)     : {z_approx:.3f}")
    print(f"p  (approx)     : {p_approx:.4f}")
    assert abs(p_exact - p_approx) < 0.01, "p-values diverge by more than 0.01"
    print("OK: both functions agree.")


if __name__ == "__main__":
    test_mannwhitneyu_4auc()
