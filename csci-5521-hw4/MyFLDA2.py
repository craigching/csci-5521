import numpy as np


def my_mean(X):
    """Returns the means of the colums of X

    """
    n = np.float64(X.shape[0])
    return X.sum(axis=0) / n


class MyFLDA2Class:

    def __init__(self, X):

        self.X = X
        N, D = self.X.shape

        self.m = my_mean(X)
        self.S = np.zeros((D, D))

        # Calculate the class scatter matrix
        for t in range(N):
            d = X[t] - self.m
            p = np.outer(d, d.T)
            self.S += p

    def project(self, w):
        p = self.X.dot(w)
        return my_mean(p)


class MyFLDA2:

    def __init__(self, threshold=None):
        self.threshold = threshold

    def fit(self, X, y):

        classes = np.unique(y)

        # Break the dataset up into unique classes
        C_1 = MyFLDA2Class(X[y == classes[0]])
        C_2 = MyFLDA2Class(X[y == classes[1]])

        # Calculate the total within class scatter
        S_w = C_1.S + C_2.S

        # Calculate the projection matrix
        self.w = np.linalg.inv(S_w).dot(C_2.m - C_1.m)

        # We've learned the transformation matrix/vector, now we need
        # to determine the classification threshold

        if self.threshold is not None:
            self.z_0 = self.threshold.dot(self.w)
        else:
            m_1 = C_1.project(self.w)
            m_2 = C_2.project(self.w)

            # Our threshold for prediction is defined to be the
            # average of the class means
            self.z_0 = (m_1 + m_2) / 2

    def predict(self, X):
        b = X.dot(self.w) >= self.z_0
        return np.where(b, 1, -1)
