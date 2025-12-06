from scipy.optimize import minimize, Bounds, LinearConstraint
import numpy as np

def compute_risk_parity_weights(cov, risk_budgets, pos_weights=True, custom_objective_func = None):
    
    # minimize SSE CSD function
    def min_sse_csd(w, cov, risk_budgets):
        vol = np.sqrt(w.T.dot(cov).dot(w))
        csd = w * (cov.dot(w)) / vol
        csd_budgeted = csd / risk_budgets
        # print(csd_budgeted.shape, risk_budgets.shape)
        return np.sum((csd_budgeted - np.mean(csd_budgeted))**2)

    # Equality constraint: sum(w) = 1 -> sum(w) -1 = 0
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # bounds
    if pos_weights:
        bounds = ((0, None),)*cov.shape[0]
    else:
        bounds = None

    # initial guess
    w0 = (np.ones((cov.shape[0], 1)) / cov.shape[0]).ravel()

    if custom_objective_func is not None:
        objective_func = lambda x: custom_objective_func(x, cov, risk_budgets)
    else:
        objective_func = lambda x: min_sse_csd(x, cov, risk_budgets)
    result = minimize(objective_func, w0, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol': 1e-20, 'maxiter': 1000, 'disp': True})

    return result.x

def inverse_volatility_weights(cov):
    stds = np.sqrt(cov.diagonal())
    over_vol_sum = np.sum(1/stds)
    weights = 1/stds * (1/over_vol_sum)
    return weights

def compute_max_sharpe_weights(means, cov, rfr, weight_bounds = (0, None)):
    
    # minimize SSE CSD function
    def objective_function(w, means, cov, rfr):
        vol = np.sqrt(w.T.dot(cov).dot(w))
        sharpe = (w.dot(means) - rfr) / vol
        return sharpe

    # Equality constraint: sum(w) = 1 -> sum(w) -1 = 0
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # bounds
    bounds = (weight_bounds,)*cov.shape[0]

    # initial guess
    w0 = (np.ones((cov.shape[0], 1)) / cov.shape[0]).ravel()
    result = minimize(lambda x: -objective_function(x, means, cov, rfr), w0, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol': 1e-14, 'maxiter': 1000, 'disp': True})

    return result.x

def efficient_frontier(means, cov, pos_weights=True, Rp=None, vol=None, custom_min_func = None):
    """
        Custom mean function must be a 3 input function, func(x, mean, cov), as this function
        doesn't innately know what needs to be passed to it, so everything is passed
    """
    if Rp is None and vol is None and custom_min_func is None:
        raise ValueError("One of returns or volatility must be given")
    
    if Rp is not None and vol is not None:
        raise ValueError("Only one of returns or volatility can be given")
    
    def var(w, cov):
        return w.T @ cov @ w
    
    def ret(w, means):
        return -np.sum(w*means)

    w_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    constraints = [w_constraint]

    if pos_weights:
        bounds = ((0, None),)*cov.shape[0]
    else:
        bounds = None
    
    if Rp is not None:
        ret_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w*means) - Rp}
        constraints.append(ret_constraint)
        min_function = lambda x: var(x,cov)
    elif vol is not None:
        vol_constraint = {'type': 'eq', 'fun': lambda w: (w.T @ cov @ w) - vol**2}
        constraints.append(vol_constraint)
        min_function = lambda x: ret(x, means)
    
    if custom_min_func is not None:
        min_function = lambda x: custom_min_func(x, means, cov)
    
    result = minimize(fun=min_function, x0=np.ones(means.shape[0])/means.shape[0], method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-20, 'maxiter': 1000, 'disp': False})
    
    return result.x

