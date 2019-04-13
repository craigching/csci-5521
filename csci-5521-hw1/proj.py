import numpy as np


def rand_proj(X, d):
    m = X.shape[1]
    mu, sigma = 0, 1
    G = np.random.normal(mu, sigma, m*d).reshape(m, d)
    return np.dot(X, G)


def quad_proj(X):
    A = np.concatenate([X, np.square(X)], axis=1)
    for col in range(X.shape[1]):
        A = np.append(A, X[:, col:col+1]*X[:, col+1:], axis=1)
    return A
