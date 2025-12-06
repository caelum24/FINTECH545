import numpy as np
from scipy.stats import t, norm
from copulae.elliptical import GaussianCopula
from copulae.elliptical import StudentCopula
import yfinance as yf
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
from scipy.optimize import minimize

yahoo_tickers = {
    # Sectors / Industries
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Materials": "XLB",
    "Real Estate": "RWR", # XLRE started in 2016
    "Communication Services": "VOX", # XLC started in 2016

    # Commodities
    "Broad Commodities": "DBC",
    "Crude Oil": "USO",
    "Gold": "GLD",
    # "Silver": "SLV",
    "Natural Gas": "UNG",
    "Agriculture": "DBA",

    # FX / Currencies
    "US Dollar Index": "UUP",
    "EUR/USD": "FXE",
    # "GBP/USD": "FXB",
    "JPY/USD": "FXY",
    # "CAD/USD": "FXC"
}

# get yfinance data
# def compute_returns_from_yahoo_data(data):
#     returns = data.pct_change().dropna()
#     return returns

def get_yfinance_data(tickers, **yf_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(tickers=tickers, **yf_kwargs)['Close'] #, period="5y",
    returns = data.pct_change().dropna()
    return data, returns


def get_breakpoints(returns, n_pts=10, model="normal", min_penalty = 10, max_penalty = 1000, tolerance=0, verbose=False) -> tuple[list, int]:
    algo = rpt.Pelt(model=model).fit(returns) # rbf

    pred_bkpts = []
    pen = max_penalty
    while not pred_bkpts or abs(len(pred_bkpts) - n_pts) > tolerance:
        if len(pred_bkpts) > n_pts:
            min_penalty = pen
            pen = (pen + max_penalty) // 2
        else:
            max_penalty = pen
            pen = (pen + min_penalty) // 2
        if verbose:
            print("Trying Pen", pen)
        *pred_bkpts, num_samples = algo.predict(pen=pen)
        if verbose:
            print("Num bkpts", len(pred_bkpts))

        if pen == max_penalty or pen == min_penalty:
            break

    return pred_bkpts, pen

def plot_breakpoints(price_data: pd.Series, pred_bkpts: list):
    plt.plot(price_data)
    plt.vlines(price_data.index[pred_bkpts], ymin=min(price_data), ymax=max(price_data), colors=["black"]*len(pred_bkpts), linestyles=["--"]*len(pred_bkpts))
    # plt.show()

def get_max_corr(mat) -> tuple[tuple[int, int], float]:
    mask = ~np.eye(mat.shape[0], dtype=bool)
    off_diag_indices = np.argwhere(mask)
    max_idx = mat[mask].argmax()
    i, j = off_diag_indices[max_idx]
    return (i, j), mat[i, j]

def construct_corr_from_rhos(dim, rho_array) -> np.array:
    upper_triangular = np.zeros((dim, dim))
    indices = np.triu_indices(dim, k=1)
    upper_triangular[indices] = rho_array
    corr = upper_triangular + upper_triangular.T + np.eye(dim)
    return corr

def compute_tail_dependence(df, corr) -> np.array:

    def t_tail_dependence(df, rho):
        # df = degrees of freedom
        # rho = correlation between variables i,j
        arg = -np.sqrt((df + 1) * (1 - rho) / (1 + rho))
        return 2 * t.cdf(arg, df + 1)

    # Example for all pairs in a fitted copula:
    tail_dep_matrix = np.zeros_like(corr)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            tail_dep_matrix[i, j] = t_tail_dependence(df, corr[i, j])
        
    return tail_dep_matrix

def transform_data_to_uniform(data, dist = "t", rv_params=None) -> tuple[np.array, dict]:
    '''
        Rows are assumed to be entries and columns are variables

        You can either fit to the existing data or pass in your own model parameters
    '''

    if dist not in ["t", "norm"]:
        raise ValueError("distribution must be normal or t")

    if dist == "t":
        model = t
    elif dist == "norm":
        model = norm
    
    fit_params = {}
    Uniform_cols = []
    for i in range(data.shape[1]):
        if rv_params is None:
            params = model.fit(data[:, i])
        
        if dist == "t":
            if rv_params is None:
                params = {"loc": params[1], "scale":params[2], "df": params[0]}
            else:
                params = rv_params[i]
            U_col = model.cdf(data[:, i], **params)
        elif dist == "norm":
            if rv_params is None:
                params = {"loc": params[0], "scale":params[1]}
            else:
                params = rv_params[i]
            U_col = model.cdf(data[:, i], **params)
        
        Uniform_cols.append(U_col)
        fit_params[i] = params

    U = np.array(Uniform_cols).T

    if rv_params is None:
        return U, fit_params
    else:
        return U, rv_params

def get_mean_returns(rv_params, scale=1):
    mean_returns = np.array([param['loc']*scale for key, param in rv_params.items()])
    return mean_returns

def get_vols(rv_params, scale=1):
    vols = np.array([param['scale']*np.sqrt(scale) for key, param in rv_params.items()])
    return vols

def transform_uniform_to_marginal(U: np.array, rv_params: dict, dist: str = "t"):
    '''
       inverse of transform data to uniform
       Rows are assumed to be entries and columns are variables
    '''

    if dist not in ["t", "norm"]:
        raise ValueError("distribution must be normal or t")

    if dist == "t":
        model = t
    elif dist == "norm":
        model = norm
    
    data_cols = []
    for i in range(U.shape[1]):
        params = rv_params[i]
        if dist == "t":
            data_col = model.ppf(U[:, i], **params)
        elif dist == "norm":
            data_col = model.ppf(U[:, i], **params)
        
        data_cols.append(data_col)

    data = np.array(data_cols).T
    return data

def simulate_from_copula(copula: GaussianCopula | StudentCopula, n_sims):
    sim_U = copula.random(n_sims)
    return sim_U

def simulate_time_scaled_returns(copula, dim, rv_params, n_sims:int, scale:int, dist="t"):
    sim_U = simulate_from_copula(copula, n_sims*scale)
    sim_returns = transform_uniform_to_marginal(sim_U, rv_params=rv_params, dist=dist)
    sim_returns_stacked = sim_returns.reshape((n_sims, scale, dim))
    scaled_returns = np.prod((1+sim_returns_stacked), axis=1) - 1
    return scaled_returns


def compute_portfolio_returns(simulated_returns: np.ndarray,
                                prices: np.ndarray,
                                holdings: np.ndarray) -> pd.Series:
    simulated_values = simulated_returns * prices + prices # assumes arithmetic returns
    total_values = simulated_values.dot(holdings)
    initial_value = prices.dot(holdings)
    portfolio_returns = (total_values - initial_value) / initial_value
    return portfolio_returns

def compute_var_es(returns, alpha=0.05):
    VaR = np.percentile(returns, 100*alpha)
    tail_mask = returns <= VaR
    ES = -np.mean(returns[tail_mask])
    return -VaR, ES

def portfolio_ES_contributions(sim_returns, weights, alpha=0.05):
    portfolio_returns = sim_returns @ weights
    VaR = np.percentile(portfolio_returns, 100*alpha)
    tail_mask = portfolio_returns <= VaR
    marg_ES = np.mean(-sim_returns[tail_mask, :], axis=0)
    contrib = weights * marg_ES
    return contrib, marg_ES, np.sum(contrib)

def compute_es_sharpe(w, means, prices, rfr, sim_returns, alpha):
    portfolio_value = 1 # value doesn't matter because we're doing percent ES
    holdings =  portfolio_value * w / prices
    portfolio_returns = compute_portfolio_returns(sim_returns, prices, holdings)
    VaR_pct, ES_pct = compute_var_es(portfolio_returns, alpha=alpha)
    sharpe = (w.dot(means) - rfr) / ES_pct
    return sharpe

def compute_optimal_es_sharpe_weights(means, prices, rfr, sim_returns, alpha = 0.05, weight_bounds = (0, None)):
    
    # Equality constraint: sum(w) = 1 -> sum(w) -1 = 0
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # bounds
    bounds = (weight_bounds,)*means.shape[0]

    # initial guess
    w0 = (np.ones((means.shape[0])) / means.shape[0])
    result = minimize(lambda x: -compute_es_sharpe(x, means, prices, rfr, sim_returns, alpha), w0, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol': 1e-14, 'maxiter': 1000, 'disp': True})

    return result.x

def compute_es_risk_parity_weights(sim_returns, risk_budgets, alpha = 0.05, weight_bounds = (0, None)):
    def min_sse_ces(w, sim_returns, risk_budgets, alpha=0.05):
        cES, _, ES = portfolio_ES_contributions(sim_returns, w, alpha=alpha)
        cES_budgeted = cES / risk_budgets
        return np.sum((cES_budgeted - np.mean(cES_budgeted))**2)

    # Equality constraint: sum(w) = 1 -> sum(w) -1 = 0
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # bounds
    bounds = (weight_bounds,)*sim_returns.shape[1]

    # initial guess
    w0 = (np.ones((sim_returns.shape[1])) / sim_returns.shape[1])
    result = minimize(lambda x: -min_sse_ces(x, sim_returns, risk_budgets, alpha), w0, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol': 1e-14, 'maxiter': 1000, 'disp': True})

    return result.x


def compute_max_drawdown(portfolio_value_series):
    max_indices = list(portfolio_value_series.index[(portfolio_value_series.cummax() == portfolio_value_series)])
    max_indices.append(portfolio_value_series.index[-1])
    max_drawdown = 0
    max_drawdown_period = None
    prev_index = None
    for index in max_indices:
        if prev_index is None:
            prev_index = index
            continue

        # because of max_indices, we know that the max will be the 1st thing and min will come after
        # this ensures we get a true max drawdown
        period_max_value = portfolio_value_series.loc[prev_index:index].max()
        period_min_value = portfolio_value_series.loc[prev_index:index].min()

        drawdown = (period_min_value - period_max_value) / period_max_value

        if drawdown < max_drawdown:
            max_drawdown = drawdown
            max_drawdown_period = (prev_index, index)
        
        prev_index = index
    
    return max_drawdown, max_drawdown_period
    












