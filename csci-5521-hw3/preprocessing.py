import numpy as np


def my_label_binarize(y, classes=None):
    '''
    Converts the data into one-hot format.  This is modeled after the
    sklearn function of similar name

    '''
    D = classes.shape[0]
    N = y.shape[0]
    onehot = np.zeros((N, D))
    y = y.astype(np.int, copy=True)
    onehot[np.arange(N), y] = 1
    return onehot


def zscale(X, m=None, s=None):
    '''
    Standardizes the data X and returns the mean and standard
    deviation calculated so the same transformation can be applied in
    predict

    '''
    # Scale the data
    if m is None:
        m = X.mean(axis=0)
    if s is None:
        s = X.std(axis=0)
    X = (X - m) / s
    return X, m, s


def fit_intercept(X):
    '''
    Adds an intercept to the design matrix X
    '''
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))
