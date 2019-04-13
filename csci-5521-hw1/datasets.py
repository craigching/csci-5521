from sklearn.datasets import load_boston, load_digits
import numpy as np


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


def prepare_digits():
    Digits = load_digits()
    X = Digits.data
    y = Digits.target
    return X, y


def print_probabilities(y):
    zeros = y.shape[0] - np.count_nonzero(y)
    ones = np.count_nonzero(y)
    p0 = zeros / float(y.shape[0])
    p1 = ones / float(y.shape[0])
    print('p(y = 1): {}, p(y = 0): {}'.format(p1, p0))
