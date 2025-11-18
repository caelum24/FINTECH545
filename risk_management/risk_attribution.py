import numpy as np

def ex_post_attribution(returns, w0):

    def carino_k(returns):
        R = np.prod(returns+1)-1
        GR = np.log(R + 1)
        K = GR / R
        k_t = np.log(1 + returns) / (K * returns)
        return k_t
    
    # GET WEIGHTS OVER TIME
    weights_over_time = [w0]
    w = w0
    for time_return in returns:
        #updating weights:
        new_w = w * (1+time_return)
        w = new_w / np.sum(new_w)
        weights_over_time.append(w)

    weights_over_time = np.array(weights_over_time)[:-1] # get rid of weights after last returns
    weighted_returns = weights_over_time * returns

    # COMPUTE TOTAL AND ATTRIBUTED RETURNS FOR EACH ASSET
    portfolio_returns = np.sum(weighted_returns, axis=1)
    k_t = carino_k(portfolio_returns)

    asset_returns = np.prod(returns+1, axis=0)-1
    total_portfolio_return = np.prod(portfolio_returns+1)-1
    
    # RETURN ATTRIBUTION
    return_attribution_over_time = (weighted_returns) * k_t[:, np.newaxis]
    attributed_returns = np.sum(return_attribution_over_time, axis=0)
    
    # VOLATILITY ATTRIBUTION
    X = np.column_stack([weighted_returns, portfolio_returns])
    cov_matrix = np.cov(X, rowvar=False)
    asset_portfolio_covs = cov_matrix[-1, :-1]
    portfolio_vol = np.sqrt(cov_matrix[-1, -1])
    risk_attributions = asset_portfolio_covs / portfolio_vol

    return  portfolio_returns, total_portfolio_return, asset_returns, k_t, weights_over_time, attributed_returns,  risk_attributions, portfolio_vol


def ex_post_attribution_of_factors(stock_returns, factor_returns, stock_factor_betas, w0):
    def carino_k(returns):
        R = np.prod(returns+1)-1
        GR = np.log(R + 1)
        K = GR / R
        k_t = np.log(1 + returns) / (K * returns)
        return k_t
    
    # figure out stock weighted returns
    stock_cum_returns = np.cumprod(1 + stock_returns, axis=0)
    numerators = np.vstack([np.ones_like(w0), stock_cum_returns[:-1]]) * w0
    stock_weights_over_time = numerators / numerators.sum(axis=1, keepdims=True)

    # COMPUTE RETURNS

    # compute portfolio returns
    stock_weighted_returns = stock_weights_over_time * stock_returns
    portfolio_returns = np.sum(stock_weighted_returns, axis=1)
    k_t = carino_k(portfolio_returns)

    # compute factor returns
    factor_derived_stock_returns = factor_returns.dot(stock_factor_betas.T)
    weighted_factor_stock_returns = factor_derived_stock_returns * stock_weights_over_time

    # compute alpha returns
    alpha_returns = portfolio_returns - np.sum(weighted_factor_stock_returns, axis=1)
    
    # get total returns
    factor_total_returns = np.prod(factor_returns+1, axis=0)-1
    alpha_total_return = np.prod(alpha_returns + 1)-1
    portfolio_total_return = np.prod(portfolio_returns + 1)-1

    # COMPUTE RETURN ATTRIBUTIONS

    # factor attributions
    factor_weights = stock_weights_over_time.dot(stock_factor_betas)
    weighted_factor_returns = factor_returns * factor_weights
    weighted_factor_geometric_returns = weighted_factor_returns * k_t[:, np.newaxis]
    factor_return_attr = np.sum(weighted_factor_geometric_returns, axis=0)

    # alpha attributions
    stock_alpha_returns = stock_returns - factor_derived_stock_returns
    weighted_stock_alpha_returns = stock_alpha_returns * stock_weights_over_time
    alpha_returns = np.sum(weighted_stock_alpha_returns * k_t[:, np.newaxis], axis=1) 
    alpha_return_attr = np.sum(alpha_returns)

    # COMPUTE VOL ATTRIBUTIONS
    X = np.column_stack([weighted_factor_returns, portfolio_returns])
    cov_matrix = np.cov(X, rowvar=False)
    asset_portfolio_covs = cov_matrix[-1, :-1]
    portfolio_vol = np.sqrt(cov_matrix[-1, -1])
    factor_risk_attr= asset_portfolio_covs / portfolio_vol
    alpha_risk_attr = portfolio_vol - np.sum(factor_risk_attr)

    return factor_total_returns, alpha_total_return, portfolio_total_return, factor_return_attr, alpha_return_attr, factor_risk_attr, alpha_risk_attr, portfolio_vol