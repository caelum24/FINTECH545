import numpy as np
from scipy.stats import norm

def European_GBSM(S, X, T, vol, r, b, option_type="call"):
    '''
        S - Underlying Price
        X - Strike Price
        T - is the Time to Maturity
        vol - is the implied volatilityσ
        r - is the risk free rate
        b - is the cost of carry

        black scholes where rf = b
    '''

    option_type = str.lower(option_type)
    if option_type not in ["call", "put"]:
        raise ValueError("Option type must be call or put")
        
    d1 = (np.log(S/X) + (b + (vol**2)/2) * T) / (vol * np.sqrt(T))

    d2 = d1 - vol * np.sqrt(T)

    if option_type == "call":
        value = S * np.exp((b-r)*T) * norm.cdf(d1) - X * np.exp(-r*T) * norm.cdf(d2)
        delta = np.exp((b-r)*T) * norm.cdf(d1)
        theta = - (S * np.exp((b-r)*T) * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) - (b-r) * S*np.exp((b-r)*T) * norm.cdf(d1) - r*X*np.exp(-r*T) * norm.cdf(d2)

        rho = T * X * np.exp(-r*T) * norm.cdf(d2)
        carry_rho = T*S * np.exp((b-r)*T) * norm.cdf(d1) # dP / db

    else:
        value = X * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp((b-r)*T) * norm.cdf(-d1)
        delta = np.exp((b-r)*T) * (norm.cdf(d1)-1)
        theta = - (S * np.exp((b-r)*T) * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) + (b-r) * S*np.exp((b-r)*T) * norm.cdf(-d1) + r*X*np.exp(-r*T) * norm.cdf(-d2)

        rho = -T * X * np.exp(-r*T) * norm.cdf(-d2)
        carry_rho = -T*S * np.exp((b-r)*T) * norm.cdf(-d1) # dP / db
    
    gamma = norm.pdf(d1) * np.exp((b-r)*T) / (S * vol * np.sqrt(T))
    vega = S * np.exp((b-r)*T) * norm.pdf(d1) * np.sqrt(T)
    
    
    return value, delta, gamma, vega, theta, rho, carry_rho


def American_Binary_Tree(S, X, T, vol, r, b, N, option_type = "call"):
    option_type = str.lower(option_type)
    if option_type not in ["call", "put"]:
        raise ValueError("Option type must be call or put")

    dt = T/N
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = np.exp(-r*dt)
    payoff_side = 1 if option_type=="call" else -1

    # vectorize for ease of use
    num_nodes = lambda N: int((N+1)*(N+2)/2) # 0 indexed (0th column to start)
    total_nodes = num_nodes(N)
    get_idx = lambda j, i: num_nodes(j-1)+i # 0 indexed

    option_values = [0.0]*total_nodes

    # work backward from future to present
    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = get_idx(j, i)
            option_values[idx] = max(0.0, payoff_side*(S*(u**i)*(d**(j-i)) - X))

            if j < N: # if not the final layer, we might early exercise
                option_values[idx] = max(option_values[idx], df * (pu*option_values[get_idx(j+1, i+1)] + pd*option_values[get_idx(j+1, i)]))
    
    return option_values[0]

def approx_delta(h, S, X, T, vol, r, b, N, option_type, richardson = True):
    def D1(h):
        v_up  = American_Binary_Tree(S + h, X, T, vol, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S - h, X, T, vol, r, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        richardson_approx = (4*D_h2 - D_h) / 3
        return richardson_approx
    
    else: 
        approx = D1(h)
        return approx

def approx_theta(h, S, X, T, vol, r, b, N, option_type, richardson = True):
    def D1(h):
        v_up  = American_Binary_Tree(S, X, T+h, vol, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S, X, T-h, vol, r, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        richardson_approx = (4*D_h2 - D_h) / 3
        return richardson_approx
    
    else: 
        approx = D1(h)
        return approx

def approx_vega(h, S, X, T, vol, r, b, N, option_type, richardson = True):
    def D1(h):
        v_up  = American_Binary_Tree(S, X, T, vol+h, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S, X, T, vol-h, r, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        richardson_approx = (4*D_h2 - D_h) / 3
        return richardson_approx
    
    else: 
        approx = D1(h)
        return approx

def approx_rho(h, S, X, T, vol, r, b, N, option_type, richardson = True):
    def D1(h):
        v_up  = American_Binary_Tree(S, X, T, vol, r+h, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S, X, T, vol, r-h, b, N=N, option_type=option_type)
        return (v_up - v_dn) / (2*h)
    
    if richardson:
        D_h = D1(h)
        D_h2 = D1(h/2)
        richardson_approx = (4*D_h2 - D_h) / 3
        return richardson_approx
    
    else: 
        approx = D1(h)
        return approx

def approx_gamma(h, S, X, T, vol, r, b, N, option_type, richardson = True):
    def D2(h):
        v_up  = American_Binary_Tree(S + h, X, T, vol, r, b, N=N, option_type=option_type)
        v_mid = American_Binary_Tree(S,     X, T, vol, r, b, N=N, option_type=option_type)
        v_dn  = American_Binary_Tree(S - h, X, T, vol, r, b, N=N, option_type=option_type)
        return (v_up - 2*v_mid + v_dn) / (h**2)
    
    if richardson:
        D_h   = D2(h)
        D_h2  = D2(h/2)
        richardson_approx = (4*D_h2 - D_h) / 3
        return richardson_approx
    else:
        approx = D2(h)
        return approx


def American_Binary_Tree_Discontinuous_Div(S, X, T, vol, r, div_amts: list, div_times: list, N, option_type="call"):

    '''
        NOTE -> div_times came in data as number of days, it has been updated here to be annualized (matching T) because it makes more sense that way
                This means we have to get the indices by dividing by dt, and we need N to be a multiple of all of our div times in order for these
                to naturally be integers without rounding (though we use rounding just in case, they should be integer values by default)

        NOTE -> we were told in class that div_amts is in dollars, but that's strange since the div amounts are 0.01, which is incredibly small
    '''
    option_type = str.lower(option_type)
    if option_type not in ["call", "put"]:
        raise ValueError("Option type must be call or put")
        
    if len(div_amts) != len(div_times):
        raise IndexError("Div amounts and times must be the same size")
    
    if not div_amts or div_times[0] > T:
        return American_Binary_Tree(S, X, T, vol, r, b=r, N=N, option_type=option_type)

    dt = T/N
    div_time_indices = [round(div_time/dt) for div_time in div_times]
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(r*dt)-d)/(u-d)
    pd = 1.0-pu
    df = np.exp(-r*dt)
    payoff_side = 1 if option_type=="call" else -1

    # vectorize for ease of use
    num_nodes = lambda N: int((N+1)*(N+2)/2) # 0 indexed (0th column to start)
    get_idx = lambda j, i: num_nodes(j-1)+i # 0 indexed
    num_divs = len(div_times)
    total_nodes = num_nodes(div_time_indices[0])

    option_values = [0.0]*total_nodes

    # work backward from future to present
    for j in range(div_time_indices[0], -1, -1):
        for i in range(j, -1, -1):
            idx = get_idx(j, i)
            price = S*(u**i)*(d**(j-i))

            if j < div_time_indices[0]:
                option_values[idx] = max(0.0, payoff_side*(price - X))
                option_values[idx] = max(option_values[idx], df * (pu*option_values[get_idx(j+1, i+1)] + pd*option_values[get_idx(j+1, i)]))
            else:
                no_exercise_val = American_Binary_Tree_Discontinuous_Div(price-div_amts[0], X, T-div_times[0], vol, r, div_amts[1:], [div_time - div_times[0] for div_time in div_times[1:]], N-div_time_indices[0], option_type)
                exercise_val = max(0, payoff_side*(price-X))
                option_values[idx] = max(exercise_val, no_exercise_val)
    
    return option_values[0]