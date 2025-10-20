import numpy as np
import pandas as pd
from scipy.stats import t 
from statsmodels.miscmodels.tmodel import TLinearModel
import statsmodels.api as sm

def fit_multivariate_normal_dist(x: pd.DataFrame):
    # columns are variables, and rows are data points

    """
        This function takes an MxN dataframe where M = number of data points and
        N = number of variables. Then, the mean and covariance matrix are computed
        and returned as a pandas dataframe
    """
    # pandas form
    mu_vector = x.mean()
    covariance_matrix = x.cov()

    # numpy form
    # mu_vector = np.mean(x, axis=0)
    # covariance_matrix = np.cov(y, rowvar=False) #unbiased by default

    return mu_vector, covariance_matrix

def fit_univariate_t_dist(x):
    """
        This function takes an Mx1 dataframe where M = number of data points and
        N = number of variables. Then, the mean and covariance matrix are computed
        and returned as a pandas dataframe
    """
    # pandas or numpy array, columns are variables
    nu, mu, sigma = t.fit(x)
    return mu, sigma, nu

def t_regression(X: pd.DataFrame, y:pd.Series, add_constant:bool = True, print_summary = False):

    """
        This function takes an MxN dataframe x where M = number of data points and
        N = the number of variables and a Mx1 series y. There is an optional add_constant
        parameter that determines whether the t-regression incorporates a constant term
    """

    if add_constant:
        X = sm.add_constant(X)
    
    model = TLinearModel(y, X)
    result = model.fit()

    if add_constant:
        alpha = result.params[0]
        betas = result.params[1:-2]
        mu = 0.0 # mean is 0 if we included a constant (intercept term)
    else:
        alpha = 0.0
        betas = result.params[0:-2]
        mu = np.mean(y - (X @ betas)) # if no intercept, estimate offset manually

    nu = result.params[-2]
    sigma = result.params[-1]


    if print_summary:
        result.summary()

    return alpha, betas, mu, sigma, nu