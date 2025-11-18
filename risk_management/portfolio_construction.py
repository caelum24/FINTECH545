from scipy.optimize import minimize, Bounds, LinearConstraint
import numpy as np

def compute_risk_parity_weights(cov, risk_budgets):
    
    # minimize SSE CSD function
    def objective_function(w, cov, risk_budgets):
        vol = np.sqrt(w.T.dot(cov).dot(w))
        csd = w * (cov.dot(w)) / vol
        csd_budgeted = csd / risk_budgets
        # print(csd_budgeted.shape, risk_budgets.shape)
        return np.sum((csd_budgeted - np.mean(csd_budgeted))**2)

    # Equality constraint: sum(w) = 1 -> sum(w) -1 = 0
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # bounds
    bounds = ((0, None),)*cov.shape[0]

    # initial guess
    w0 = (np.ones((cov.shape[0], 1)) / cov.shape[0]).ravel()
    result = minimize(lambda x: objective_function(x, cov, risk_budgets), w0, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol': 1e-20, 'maxiter': 1000, 'disp': True})

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

