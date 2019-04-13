import numpy as np
import numpy.linalg as la


def my_mean(X):
    """Returns the means of the colums of X

    """
    n = np.float64(X.shape[0])
    return X.sum(axis=0) / n


def my_cov(X):
    """Returns the sample covariance matrix for X

    """
    mu = np.ones(np.shape(X)) * my_mean(X)
    n = np.float64(np.shape(X)[0])
    return (1 / (n - 1)) * (X - mu).T.dot(X - mu)


def my_var(X):
    """Not implemented since not required for this project

    """
    pass


class QuadraticGaussianDiscriminant:
    """Quadratic discriminant for Multivariate Gaussian classification

    Parameters
    ----------
        name : {string}
            The label of our class

        X : {array-like, sparse matrix}
            Class-specific training data from which discriminant parameters
            will be estimated

        n : {number}
            The total number of observations in the whole dataset
    """
    def __init__(self, name, X, n):
        # Keep our class label
        self.name = name

        # Because we're required to, intialize covariance and means
        S = np.identity(X.shape[1])
        means = np.zeros(X.shape[1])

        # Now properly initalize S and means
        S = my_cov(X)
        means = my_mean(X)

        # Estimate the prior from our data
        prior = X.shape[0] / np.float64(n)

        # Calculate the inverse and determinant of S for later use
        Sinv = la.inv(S)
        Sdet = la.det(S)

        # Quadratic discriminant ref Alpaydin "Introdution to Machine
        # Learning, Third Edition"
        self.W_i = (-1/2.) * Sinv
        self.w_i = Sinv.dot(means)
        self.w_i0 = (-1/2.) * (
            means.T.dot(
                Sinv.dot(
                    means))) - (1/2.) * np.log(
                        Sdet) + np.log(prior)

    def discriminant(self, X):
        """Evaluate the learned discriminant function for the given
        observations

    Parameters
    ----------
        X : {array-like, sparse matrix}
            Observations on which to evaluate the discriminant function

    Returns
    -------
        score : float
            Returns the value evaluated by the discriminant function
        """
        return X.T.dot(
            self.W_i.dot(
                X)) + self.w_i.T.dot(
                    X) + self.w_i0


class LinearGaussianDiscriminant:
    """Linear discriminant for Multivariate Gaussian classification

    Parameters
    ----------
        name : {string}
            The label of our class

        X : {array-like, sparse matrix}
            Class-specific training data from which discriminant parameters
            will be estimated

        n : {number}
            The total number of observations in the whole dataset

        S : {array-like}
            The shared covariance matrix estimated from the full data set
    """
    def __init__(self, name, X, n, S):
        # Keep our class label
        self.name = name

        # Initialize means, S is provided
        means = my_mean(X)

        # Estimate the prior from our data
        prior = X.shape[0] / np.float64(n)

        # Invert S for later use
        Sinv = la.inv(S)

        # Linear discriminant
        self.w_i = Sinv.dot(means)
        self.w_i0 = (-1/2.) * (
            means.T.dot(Sinv.dot(
                means))) + np.log(prior)

    def discriminant(self, X):
        """Evaluate the learned discriminant function for the given
        observations

    Parameters
    ----------
        X : {array-like, sparse matrix}
            Observations on which to evaluate the discriminant function

    Returns
    -------
        score : float
            Returns the value evaluated by the discriminant function
        """
        return self.w_i.T.dot(X) + self.w_i0


class MultiGaussClassify:
    """Multivariate Gaussian classifier

    Parameters
    ----------
        linear : {boolean}
            If true, compute a shared covariance matrix from the whole
            of the training data and use linear discriminants
    """
    def __init__(self, linear=False):
        # Where we keep the discriminants
        self.classes = []
        # Flag to use linear discriminants if True
        self.linear = linear

    def fit(self, X, y):
        # Create new discriminants for each fit
        self.classes = []

        # Initalize the number of observations
        n = X.shape[0]

        # By default, don't calculate a common covariance marix
        S_common = None

        # In this case, we want a common covariance matrix
        if self.linear:
            S_common = my_cov(X)

        # Get the classes from the data
        classes = np.unique(y)
        ncols = X.shape[1]

        # Create and initalize one discriminant per class
        for c in classes:
            # Separate out the data for this class
            X_class = X[np.where(y == c)[0]]
            X_class.reshape(X_class.shape[0], ncols)
            discriminant = None

            # Use a linear discriminant if we have a common covariance
            # matrix
            if S_common is not None:
                discriminant = LinearGaussianDiscriminant(
                    c, X_class, n, S_common)
            else:
                discriminant = QuadraticGaussianDiscriminant(c, X_class, n)

            self.classes.append(discriminant)

    def predict(self, X):
        """Predict labels from our discrimants for the given observations

        Parameters
        ----------
        X : {array-like}
            Observations for which to predict a class

        Returns
        -------
        ypred : {array-like}
            Returns the predictions for the given observations.
        """
        ypred = []

        for i in np.arange(X.shape[0]):
            scores = []
            # Evaluate the discrimant for each class
            for cls in self.classes:
                s = cls.discriminant(X[i].T)
                scores.append(s)
            # Find the highest score
            i = np.argmax(scores)
            # Return the label for the class with the highest score
            ypred.append(self.classes[i].name)

        return np.array(ypred)
