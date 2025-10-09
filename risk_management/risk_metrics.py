import pandas as pd
import numpy as np
from scipy.stats import norm, t
from .distributions import fit_normal_dist_from_data


def univariate_normal_VaR(x: pd.DataFrame, alpha = 0.05):
    mu_vector, covariance_matrix = fit_normal_dist_from_data(x)
    mean, std = mu_vector.iloc[0], np.sqrt(covariance_matrix.iloc[0,0])

    abs_VaR = -norm.ppf(alpha, loc = mean, scale = std)
    rel_VaR = mean - norm.ppf(alpha, loc = mean, scale = std)

    return abs_VaR, rel_VaR

def univariate_t_VaR(x: pd.DataFrame, alpha = 0.05):
    nu, mu, sigma = t.fit(x)

    abs_VaR = -t.ppf(alpha, df = nu, loc = mu, scale = sigma)
    rel_VaR = mu - t.ppf(alpha, df = nu, loc = mu, scale = sigma)
        
    return abs_VaR, rel_VaR

