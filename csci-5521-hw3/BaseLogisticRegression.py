import numpy as np
from preprocessing import (
    my_label_binarize,
    zscale,
    fit_intercept
)
from optimize import batch_gradient_descent


class BaseLogisticRegression():
    """
    Base implementation of logistic regression, used by both the
    two-class and multi-class implementations.

    """

    def __init__(self):
        pass

    def _train(self, X, y, reg, iterations, eta, make_loss_grad):
        classes = np.unique(y)
        K = classes.shape[0]

        # Convert y to one-hot encoding if multi class
        if K > 2:
            if self.verbose:
                print('binarizing data')
            y = my_label_binarize(y, classes=classes)

        # Scale the data
        if self.scale:
            if self.verbose:
                print('scaling')
            X, self.m, self.s = zscale(X)

        # Add an intercept
        if self.intercept:
            if self.verbose:
                print('fitting intercept')
            X = fit_intercept(X)

        N, D = X.shape

        if K > 2:
            w = np.zeros((D, K))
        else:
            w = np.zeros(D)

        self.w = batch_gradient_descent(
            w,
            iterations,
            eta,
            make_loss_grad(X, y, reg),
            verbose=self.verbose,
        )
        if self.verbose:
            print('=== done ===')

    def _prep_predict(self, X):

        X = np.copy(X)

        if self.scale:
            X = zscale(X, self.m, self.s)[0]

        if self.intercept:
            X = fit_intercept(X)

        return X
