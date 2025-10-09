import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.miscmodels.tmodel import TLinearModel
import statsmodels.api as sm

def fit_normal_dist_from_data(x: pd.DataFrame):
    # columns are variables, and rows are data points

    # pandas form
    mu_vector = x.mean()
    covariance_matrix = x.cov()

    # mu_vector = np.mean(x, axis=0)
    # covariance_matrix = np.cov(y, rowvar=False) #unbiased by default

    return mu_vector, covariance_matrix

def fit_t_dist(x):
    # pandas or numpy array, columns are variables
    nu, mu, sigma = t.fit(x)
    return mu, sigma, nu

def t_regression(X: pd.DataFrame, y:pd.Series, add_constant:bool = True, print_summary = False):

    if add_constant:
        X = sm.add_constant(X)
    
    model = TLinearModel(y, X)
    result = model.fit()

    alpha = result.params[0]
    betas = result.params[1:-2]
    nu = result.params[-2]
    sigma = result.params[-1]
    mu = 0.0 # mean is 0 if we included a constant (intercept term)

    if print_summary:
        result.summary()

    return alpha, betas, mu, sigma, nu