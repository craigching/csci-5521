import numpy as np
from BaseLogisticRegression import BaseLogisticRegression


def softmax(z):
    # NOTE I found the "max" trick below documented in numerous places
    # on the web.  This definitely helped with numerical stability
    z -= np.max(z)
    scores = np.exp(z)
    return scores / np.sum(scores, axis=1, keepdims=True)


def multinomial_loss_grad(w, X, y, reg):
    '''Provides the loss and gradient to the train implementation for
    gradient descent

    '''

    # Number of samples
    N = X.shape[0]

    # Class probabilities
    z = X.dot(w)
    p = softmax(z)

    # Find regularized loss based on class probabilities
    loss = (-1 / N) * np.sum(y * np.log(p)) + (reg / 2) * np.sum(w * w)

    # Compute regularized gradient for loss
    # TODO We shouldn't include regularization in the intercept term
    grad = (-1 / N) * X.T.dot(y - p) + reg * w

    return loss, grad


def make_loss_grad(X, y, reg):
    '''A factory function used by gradient descent to get the loss and
    gradient

    '''
    def loss_grad(w):
        return multinomial_loss_grad(w, X, y, reg)
    return loss_grad


class MyLogisticRegGen(BaseLogisticRegression):
    """
    Parameters
    ----------
        intercept : {boolean}
            Whether to fit an intercept or not

        scale : {boolean}
            Whether to standardize the data or not

        verbose : {boolean}
            If true, produces debug output
    """

    def __init__(self, intercept=True, scale=False, verbose=True):
        self.intercept = intercept
        self.scale = scale
        self.verbose = verbose
        np.seterr(over='raise')

    def fit(self, X, y):

        reg = 1.0
        iterations = 1000
        eta = 0.00001

        self._train(X, y, reg, iterations, eta, make_loss_grad)

    def predict(self, X):

        X = self._prep_predict(X)

        preds = softmax(X.dot(self.w))

        return preds.argmax(axis=1)
