import numpy as np
import pandas as pd

def near_psd(A: pd.DataFrame, epsilon = 0.0):

    A = np.asarray(A)
    n = A.shape[0]

    invSD = None
    out = A.copy()

    diag_vals = np.diag(out)
    count_ones = np.sum(np.isclose(diag_vals, 1.0))

    if count_ones != n:
        # convert covariance matrix to correlation matrix
        stds = np.sqrt(diag_vals)
        invSD = np.diag(1.0 / stds)
        out = invSD @ out @ invSD

    # out, invSD = convert_covariance_to_correlation(out, n)

    # svd
    eigenvalues, eigenvectors  = np.linalg.eigh(out)
    eigenvalues = np.maximum(eigenvalues, epsilon)

    T = 1.0 / ((eigenvectors ** 2) @ eigenvalues)
    T = np.diag(np.sqrt(T))

    l = np.diag(np.sqrt(eigenvalues))

    B = T @ eigenvectors @ l

    out = B @ B.T

    # Add back the variance if invSD was set earlier
    if invSD is not None:
        stds = 1.0 / np.diag(invSD)
        SD = np.diag(stds)
        out = SD @ out @ SD
        # out = invSD @ out @ invSD
    
    return out


def higham_covariance(A: pd.DataFrame, tolerance = 1e-8, max_iterations= 100_000):

    def P_u(A):
        # TODO -> could add weights
        np.fill_diagonal(A, 1)
        return A
    
    def P_s(A):
        # TODO -> could add weights
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        diag = np.maximum(np.diag(eigenvalues), 0)
        A_proj = eigenvectors @ diag @ eigenvectors.T
    
        return A_proj
    
    def valid_correlation_matrix(A, tolerance):
        # check 1 on diagonals
        if not np.allclose(np.diag(A), 1):
            return False

        # check symmetry
        if not np.allclose(A, A.T, atol=tolerance):
            return False

        # ensure non-negative eigenvalues
        eigvalues = np.linalg.eigvalsh(A)
        return np.all(eigvalues >= -tolerance)

    # convert to correlation
    A = np.asarray(A)
    n = A.shape[0]

    invSD = None
    out = A.copy()

    diag_vals = np.diag(out)
    count_ones = np.sum(np.isclose(diag_vals, 1.0))

    if count_ones != n:
        # convert covariance matrix to correlation matrix
        stds = np.sqrt(diag_vals)
        invSD = np.diag(1.0 / stds)
        out = invSD @ out @ invSD
    
    # start higham
    delta_S = np.zeros_like(out)
    Y = out.copy()
    # gamma = np.inf

    for i in range(max_iterations):
        R = Y - delta_S
        X = P_s(R)
        delta_S = X - R
        Y = P_u(X)
    
        if valid_correlation_matrix(Y, tolerance):
            break
    
    # convert back to covariance if needed
    if invSD is not None:
        stds = 1.0 / np.diag(invSD)
        SD = np.diag(stds)
        Y = SD @ Y @ SD
        # out = invSD @ out @ invSD
    
    return Y #, i