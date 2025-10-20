import pandas as pd
import numpy as np
from numpy.linalg import cholesky

def compute_correlation(x:pd.DataFrame, method="pearson", drop_missing = False, exponentially_weighted = False, lambda_ = 0.97, ddof: int =1):
    """
        This method takes in an MxN dataframe where M = number of entries (rows) and
        N = number of variables. A method is chosen and whether or not we will drop
        all rows with missing values or just drop pairwise values. 

        x: our data
        method: correlation method
        drop_missing: True if drop all na rows, False if pairwise drop
        exponentially_weighted: True if correlation is exponentially weighted
        lambda: weighting for exponential weighting
    """
    
    # drop missing drops all rows with missing data
    # otherwise, we compute covariance pairwise, dropping only pairwise missing values
    
    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("Correlation method not supported")

    if drop_missing:
       x = x.dropna()
    
    if exponentially_weighted:
        ewm_corrs_over_time = x.ewm(alpha = (1-lambda_),).corr(bias=True, method=method)
        last_ewm_corr_matrix = ewm_corrs_over_time.loc[ewm_corrs_over_time.index.get_level_values(0).max()]
        return last_ewm_corr_matrix
    else:
        return x.corr(method=method, ddof=ddof) # default ddof = 1 (unbiased)

def compute_covariance(x:pd.DataFrame, drop_missing = False, exponentially_weighted = False, lambda_ = 0.97, ddof:int=1):
    """
        This method takes in an MxN dataframe where M = number of entries (rows) and
        N = number of variables. A method is chosen and whether or not we will drop
        all rows with missing values or just drop pairwise values. 

        x: our data
        method: correlation method
        drop_missing: True if drop all na rows, False if pairwise drop
        exponentially_weighted: True if correlation is exponentially weighted
        lambda: weighting for exponential weighting
    """

    # drop missing drops all rows with missing data
    # otherwise, we compute covariance pairwise, dropping only pairwise missing values
    
    if drop_missing:
       x = x.dropna()
    
    if exponentially_weighted:
        ewm_covs_over_time = x.ewm(alpha = (1-lambda_),).cov(bias=True)
        last_ewm_cov_matrix = ewm_covs_over_time.loc[ewm_covs_over_time.index.get_level_values(0).max()]
        return last_ewm_cov_matrix
    else:
        return x.cov(ddof=ddof)

def compute_covariance_with_ew_corr(x: pd.DataFrame, corr_lambda, var_lambda):

    """
        Use the EWM correlations along with the EWM variances of our variables
        to discern the EWM covariances of our data
    
    """
    ewm_var = x.ewm(alpha = (1-var_lambda),).var(bias=True)
    std_devs = np.sqrt(ewm_var.iloc[-1])
    std_dev_products_matrix = np.outer(std_devs, std_devs)

    corr = compute_correlation(x, exponentially_weighted=True, lambda_=corr_lambda)
    cov = corr*std_dev_products_matrix

    return cov


def near_psd(A: pd.DataFrame, epsilon = 0.0):
    # convert a non-psd matrix into its closest psd neighbor
    A = np.asarray(A)
    n = A.shape[0]

    invSD = None
    out = A.copy()

    diag_vals = np.diag(out)
    count_ones = np.sum(np.isclose(diag_vals, 1.0))

    if count_ones != n:
        # convert covariance matrix to correlation matrix
        stds = np.sqrt(diag_vals)
        invSD = np.diag(1.0 / stds)
        out = invSD @ out @ invSD

    # svd
    eigenvalues, eigenvectors  = np.linalg.eigh(out)
    eigenvalues = np.maximum(eigenvalues, epsilon)

    T = 1.0 / ((eigenvectors ** 2) @ eigenvalues)
    T = np.diag(np.sqrt(T))

    l = np.diag(np.sqrt(eigenvalues))

    B = T @ eigenvectors @ l

    out = B @ B.T

    # Add back the variance if invSD was set earlier
    if invSD is not None:
        stds = 1.0 / np.diag(invSD)
        SD = np.diag(stds)
        out = SD @ out @ SD
        # out = invSD @ out @ invSD
    
    return out

def higham_psd(A: pd.DataFrame, tolerance = 1e-8, max_iterations= 100_000):
    # use higham to convert non-psd matrix into a psd neighbor
    def P_u(A):
        # TODO -> could add weights
        np.fill_diagonal(A, 1)
        return A
    
    def P_s(A):
        # TODO -> could add weights
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        diag = np.maximum(np.diag(eigenvalues), 0)
        A_proj = eigenvectors @ diag @ eigenvectors.T
    
        return A_proj
    
    def valid_correlation_matrix(A, tolerance):
        # check 1 on diagonals
        if not np.allclose(np.diag(A), 1):
            return False

        # check symmetry
        if not np.allclose(A, A.T, atol=tolerance):
            return False

        # ensure non-negative eigenvalues
        eigvalues = np.linalg.eigvalsh(A)
        return np.all(eigvalues >= -tolerance)

    # convert to correlation
    A = np.asarray(A)
    n = A.shape[0]

    invSD = None
    out = A.copy()

    diag_vals = np.diag(out)
    count_ones = np.sum(np.isclose(diag_vals, 1.0))

    if count_ones != n:
        # convert covariance matrix to correlation matrix
        stds = np.sqrt(diag_vals)
        invSD = np.diag(1.0 / stds)
        out = invSD @ out @ invSD
    
    # start higham
    delta_S = np.zeros_like(out)
    Y = out.copy()
    gamma = np.inf

    for i in range(max_iterations):
        R = Y - delta_S
        X = P_s(R)
        delta_S = X - R
        Y = P_u(X)
    
        if valid_correlation_matrix(Y, tolerance):
            break
    
    # convert back to covariance if needed
    if invSD is not None:
        stds = 1.0 / np.diag(invSD)
        SD = np.diag(stds)
        Y = SD @ Y @ SD
        # out = invSD @ out @ invSD
    
    return Y #, i


def cholesky_factor(x:pd.DataFrame):
    chol = pd.DataFrame(cholesky(x))
    return chol


def compute_returns(x: pd.DataFrame, return_type = "arithmetic"):
    if return_type not in ["arithmetic", "log"]:
        raise ValueError(f"Return type must be one of {' , '.join(['arithmetic', 'log'])}")

    if return_type == "arithmetic":
        return x.pct_change().dropna()
    
    if return_type == "log":
        return np.log(x / x.shift(1)).dropna()