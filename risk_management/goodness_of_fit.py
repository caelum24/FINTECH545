from sklearn.metrics import r2_score
import numpy as np

def r2(actual: np.ndarray, predicted: np.ndarray):
    """ R2 Score """
    return r2_score(actual, predicted)

def adjr2(actual: np.ndarray, predicted: np.ndarray, rowcount: int, featurecount: int):
    """ R2 Score """
    return 1-(1-r2(actual,predicted))*(rowcount-1)/(rowcount-featurecount)

def log_likelihood(pdf, parameters, data):
    return np.sum(np.log(pdf(data, parameters)))

# lower AIC and BIC is generally preferred
def aic(k: int, L: float):
    return 2*k- 2*np.log(L)

def aicc(k: int, L: float, n: int):
    return 2*k*(n/(n-k-1)) - 2*np.log(L)

def bic(k: int, L: float, n: int):
    return k * np.log(n) - 2 * np.log(L)

# def bicc(k: int, L: float, n: int):
#     return 