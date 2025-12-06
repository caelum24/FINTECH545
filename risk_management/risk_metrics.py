import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.integrate import quad
from .distributions import fit_multivariate_normal_dist, fit_univariate_t_dist


# def univariate_normal_VaR(x: pd.DataFrame, alpha = 0.05):
    # mu_vector, covariance_matrix = fit_multivariate_normal_dist(x)
    # mean, std = mu_vector.iloc[0], np.sqrt(covariance_matrix.iloc[0,0])

def univariate_normal_VaR(mean: float, std: float, alpha = 0.05):
    """
        Take in a previously computed mean and standard deviation
        from some data and an alpha value
    """ 
    abs_VaR = -norm.ppf(alpha, loc = mean, scale = std) # TODO -> pretty sure I flipped these, as abs should assume 0 mean
    rel_VaR = mean - norm.ppf(alpha, loc = mean, scale = std)

    return abs_VaR, rel_VaR

# def univariate_t_VaR(x: pd.DataFrame, alpha = 0.05):
    # took in x, alpha
    # nu, mu, sigma = t.fit(x)
def univariate_t_VaR(mu: float, sigma: float, nu: float, alpha: float = 0.05):
    abs_VaR = -t.ppf(alpha, df = nu, loc = mu, scale = sigma)
    rel_VaR = mu - t.ppf(alpha, df = nu, loc = mu, scale = sigma)
        
    return abs_VaR, rel_VaR

# def expected_shortfall_normal(x: pd.DataFrame, alpha=0.05):

#     abs_VaR, rel_VaR = univariate_normal_VaR(x, alpha = alpha)
#     mu_vector, cov = fit_multivariate_normal_dist(x)
#     mu = mu_vector.iloc[0]
#     sigma = np.sqrt(cov.iloc[0,0])
def expected_shortfall_normal(mu:float, sigma: float, alpha = 0.05):

    # computational method using integration
    # def ev(x, mu, sigma):
    #     return x * norm.pdf(x, loc=mu, scale=sigma)
    # result, error = quad(lambda x: ev(x, mu, sigma), -np.inf, -abs_VaR)

    # abs_ES = -1/alpha * result
    # diff_ES = -(-abs_ES - mu)

    z_alpha = norm.ppf(alpha)
    VaR = mu + sigma * z_alpha
    ES = mu - sigma * norm.pdf(z_alpha) / alpha
    abs_ES = -ES
    diff_ES = -(ES - mu) # assumes 0 mean? -> I may have abs ES and diff ES flipped TODO

    # delta VaR es
    # quantile = norm.ppf(alpha, loc = mu, scale = sigma)
    # expected_shortfall = -mu + sigma * norm.pdf(quantile, loc=mu, scale=sigma)

    return abs_ES, diff_ES

# def expected_shortfall_t(x: pd.DataFrame, alpha=0.05):
#     abs_VaR, rel_VaR = univariate_t_VaR(x, alpha = alpha)
#     mu, sigma, nu = fit_univariate_t_dist(x)
def expected_shortfall_t(mu: float, sigma: float, nu: float, alpha: float = 0.05):

    # def ev(x, mu, sigma, nu):
    #     return x * t.pdf(x, loc=mu, scale=sigma, df=nu)
    # result, error = quad(lambda x: ev(x, mu, sigma, nu), -np.inf, -abs_VaR)

    # abs_ES = -1/alpha * result
    # diff_ES = -(-abs_ES - mu)

    # delta VaR es
    # quantile = norm.ppf(alpha, loc = mu, scale = sigma)
    # expected_shortfall = -mu + sigma * norm.pdf(quantile, loc=mu, scale=sigma)

    if nu <= 1:
        raise ValueError("Degrees of freedom nu must be > 1 for finite mean")

    # quantile of standard t
    t_alpha = t.ppf(alpha, df=nu)
    VaR = mu + sigma * t_alpha

    # PDF of standard t at quantile
    pdf_alpha = t.pdf(t_alpha, df=nu)

    # ES formula
    ES = mu - sigma * (nu + t_alpha**2) / (nu - 1) * pdf_alpha / alpha
    abs_ES = -ES
    diff_ES = -(ES - mu)

    return abs_ES, diff_ES


def delta_normal_var(asset_prices, weights, underlying_prices, asset_underlying_price_indices, deltas, cov, alpha = 0.05):
    portfolio_value = weights @ asset_prices

    # d asset return / d underlying return (often 1 if asset is the underlying)
    dRA_dri = deltas * underlying_prices[asset_underlying_price_indices] / asset_prices

    # how much each underlying contributes to the portfolio returns
    weight_contributions = np.zeros((underlying_prices.shape[0], weights.shape[0]))
    col_indices = np.arange(weights.shape[0])
    weight_contributions[asset_underlying_price_indices, col_indices] = weights

    grad_r = weight_contributions @ dRA_dri # dR / dri -> d portfolio return / d underlying return

    sigma_p = np.sqrt(grad_r.T @ cov @ grad_r)
    VaR = - portfolio_value * norm.ppf(alpha) * sigma_p

    return VaR


def delta_normal_es(asset_prices, weights, underlying_prices, asset_underlying_price_indices, deltas, cov, alpha=0.05):
    portfolio_value = weights @ asset_prices

    # d asset return / d underlying return (often 1 if asset is the underlying)
    dRA_dri = deltas * underlying_prices[asset_underlying_price_indices] / asset_prices

    # how much each underlying contributes to the portfolio returns
    weight_contributions = np.zeros((underlying_prices.shape[0], weights.shape[0]))
    col_indices = np.arange(weights.shape[0])
    weight_contributions[asset_underlying_price_indices, col_indices] = weights

    grad_r = weight_contributions @ dRA_dri # dR / dri -> d portfolio return / d underlying return

    # zero-mean ES closed form
    z = norm.ppf(alpha)
    pdf_z = norm.pdf(z)
    sigma_p = np.sqrt(grad_r.T @ cov @ grad_r)

    ES = portfolio_value * sigma_p * pdf_z / alpha

    return ES



