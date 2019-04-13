from sklearn.datasets import load_boston, load_digits
import numpy as np


def add_noise(X, mu=0, sigma=1):
    """Adds gaussian noise to the given matrix X
    """
    G = np.random.normal(mu, sigma, X.size).reshape(X.shape)
    return X + G


def prepare_boston(pct):
    boston = load_boston()
    y = boston.target
    X = boston.data
    p = np.percentile(y, pct)
    y1 = np.where(y >= p, 1, 0)
    return X, y1


def prepare_boston50():
    return prepare_boston(50)


def prepare_boston75():
    return prepare_boston(75)


def prepare_digits(want_noise=True):
    Digits = load_digits()
    X = Digits.data
    y = Digits.target

    # Add some gaussian noise to avoid singular covariance matrices
    if want_noise:
        X = add_noise(X, sigma=0.001)

    return X, y
