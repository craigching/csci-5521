import numpy as np
from BaseLogisticRegression import BaseLogisticRegression


def sigmoid(z):
    # NOTE: I found this in the scikit-learn implementation.  On
    # testing, it appears to give equivalent results as our canonical
    # sigmoid function and is numerically stable.
    return .5 * (1 + np.tanh(.5 * z))


def logistic_loss_grad(w, X, y, reg):

    z = X.dot(w)
    p = sigmoid(z)

    loss = -np.sum(y * z - np.log(1 + np.exp(z))) + (reg / 2) * np.sum(w * w)
    # TODO We shouldn't include regularization in the intercept term
    grad = X.T.dot(p - y) + reg * w

    return loss, grad


def make_loss_grad(X, y, reg):
    def loss_grad(w):
        return logistic_loss_grad(w, X, y, reg)
    return loss_grad


class MyLogisticReg2(BaseLogisticRegression):
    """
    Implements a concrete implementation of the two-class logistic
    regression algorithm.  This class is basically a wrapper around
    BaseLogisticRegression

    Parameters
    ----------
        intercept : {boolean}
            Whether to fit an intercept or not

        scale : {boolean}
            Whether to standardize the data or not

        verbose : {boolean}
            If true, produces debug output

    """

    def __init__(self, intercept=True, scale=True, verbose=False):
        self.intercept = intercept
        self.scale = scale
        self.verbose = verbose
        np.seterr(over='raise')

    def fit(self, X, y):

        reg = 1.0
        iterations = 500
        eta = 0.01

        self._train(X, y, reg, iterations, eta, make_loss_grad)

    def predict(self, X):

        X = self._prep_predict(X)

        ypred = []
        for i in np.arange(X.shape[0]):
            yhat = sigmoid(self.w.T.dot(X[i]))
            if yhat > 0.5:
                ypred.append(1)
            else:
                ypred.append(0)

        return np.array(ypred)
