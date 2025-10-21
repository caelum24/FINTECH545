import numpy as np
import pandas as pd
from .distributions import fit_multivariate_normal_dist
from scipy.stats import norm
from .measurements import compute_correlation
# from measurements import higham_psd, near_psd -> fix methods


def normal_monte_carlo_simulation(mean_vector, covariance_matrix, n_sims, fix_method, seed=1234):
    # simulate covariance based on a mean and covariance input

    np.random.seed(seed=seed)

    # check for positive-semidefiniteness
    eigvals = np.linalg.eigvalsh(covariance_matrix)
    if np.any(eigvals < 0):
        # if not positive-semidefinite, use fix_method to fix
        input_cov = covariance_matrix
        covariance_matrix = fix_method(input_cov)

    simulation_data = np.random.multivariate_normal(mean_vector, covariance_matrix, n_sims).T # len(cov), n
    # sim_cov = np.cov(simulation_data, bias=False)
    
    # return sim_cov
    return simulation_data

def pca_monte_carlo_simulation(mean_vector, covariance_matrix, n_sims, explained_threshold = 0.99, seed=1234):
    # simulate covariance based on pca reduced system

    np.random.seed(seed)
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    eigvals = np.clip(eigvals, 0, None)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = len(eigvals)
    pct_explained = eigvals[:k-1].sum() / eigvals.sum()
    while pct_explained > explained_threshold:
        k -= 1
        pct_explained = eigvals[:k-1].sum() / eigvals.sum()

    L = eigvecs[:,:k] @ np.diag(np.sqrt(eigvals[:k]))
    simulation_data = np.random.multivariate_normal(np.zeros(k), np.identity(k), n_sims).T # len(cov), n
    transformed_data = L @ simulation_data + mean_vector[:,np.newaxis]

    # sim_cov = np.cov(transformed_data, bias=False)
    # return sim_cov
    return transformed_data

def monte_carlo_VaR_sim(mean_vector, covariance_matrix, current_prices, holdings, n_draws, return_type = "arithmetic", alpha = 0.05, seed = 1234):

    # simulate the VaR of a system
    
    if return_type not in ["arithmetic", "geometric", "brownian"]:
        raise ValueError("Returns must be one of arithmetic, geometric, brownian")

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")
        
    if len(mean_vector) != covariance_matrix.shape[0]:
        raise ValueError("Mean matrix length must match Covariance matrix dimensions")

    rng = np.random.default_rng(seed)

    portfolio_value = current_prices.dot(holdings)

    simulated_returns = rng.multivariate_normal(mean_vector, covariance_matrix, n_draws)
    # simulated_returns = np.random.normal(mean_vector.iloc[0], np.sqrt(cov.iloc[0,0]), size = n_draws)[:, np.newaxis]

    if return_type == "arithmetic":
        simulated_prices = (1 + simulated_returns) * current_prices
    elif return_type == "geometric":
        simulated_prices = current_prices * np.exp(simulated_returns)
    elif return_type == "brownian":
        simulated_prices = current_prices + simulated_returns

    sim_portfolio_values = simulated_prices.dot(holdings)
    sorted_values = np.sort(sim_portfolio_values) # TODO -> not sure if I need this

    percentile_portfolio = np.percentile(sorted_values, 100 * alpha)
    abs_VaR = portfolio_value - percentile_portfolio
    rel_VaR = np.mean(sorted_values) - percentile_portfolio
    return abs_VaR, rel_VaR


def VaR_ES_2_level_sim_from_copula(sample_data: pd.DataFrame, holdings: np.array, prices: np.array, fix_method, n_sims = 100_000, alpha=0.05, seed=1234):

    if prices.shape != holdings.shape or len(sample_data.columns) != prices.shape[0]:
        raise ValueError("Data columns, holdings, and prices must all contain data for the sam amount of assets")

    means, cov = fit_multivariate_normal_dist(sample_data)

    quantile_vectors = pd.DataFrame()

    for i, column in enumerate(sample_data.columns):
        mean = means.iloc[i]
        sigma = np.sqrt(cov.iloc[i,i])
        # print(i, column, mean, sigma)
        col_data = sample_data.loc[:, column]
        U_vector = norm.cdf(col_data, loc=mean, scale=sigma) # get U from observations
        Z_vector = norm.ppf(U_vector, loc=0, scale = 1) # get Z from U vectors (not needed if using spearman correlation)
        quantile_vectors[column] = Z_vector

    corr = compute_correlation(quantile_vectors, method="spearman")
    # print("CORR", corr)
    simulated_Zs = normal_monte_carlo_simulation(mean_vector=np.zeros(len(means)), covariance_matrix=corr, n_sims=n_sims, fix_method=fix_method, seed=seed)
    simulated_results = pd.DataFrame()

    for i, column in enumerate(sample_data.columns):

        col_data = simulated_Zs.T[:, i]
        U_vector = norm.cdf(col_data, loc=0, scale=1)

        # get original distribution back
        mean = means.iloc[i]
        sigma = np.sqrt(cov.iloc[i,i])
        F_vector = norm.ppf(U_vector, loc=mean, scale = sigma)

        simulated_results[column] = F_vector

    simulated_results["Total"] = ((simulated_results.dot(holdings*prices)) / holdings.dot(prices))

    risk_results = []
    for i, column in enumerate(simulated_results.columns):
        if column == "Total":
            invest_value = prices.dot(holdings)
        else:
            invest_value = prices[i]*holdings[i]

        sim_size = len(simulated_results[column])
        index = int(np.floor(alpha*sim_size))
        sorted_col = simulated_results[column].sort_values().reset_index(drop=True)

        # VaR 95% is var expressed as a percentage of initial investment
        # VaR95 / initial portfolio value = VaR 95_Pct

        VaR_95pct = -sorted_col.iloc[index]
        VaR95 = VaR_95pct * invest_value
        ES_95pct = -sorted_col[:index+1].mean()
        ES95 = ES_95pct * invest_value
        risk_results.append({
                    "Stock": column,
                    "VaR95": VaR95,
                    "ES95": ES95,
                    "VaR95_Pct": VaR_95pct,
                    "ES95_Pct": ES_95pct
                }
        )
    
    return pd.DataFrame(risk_results)