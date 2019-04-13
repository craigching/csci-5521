import numpy as np
import random


def randomize(X, y):
    """Combine X and y and randomly shuffle the data"""
    data = np.append(X, y, axis=1)
    np.random.shuffle(data)
    return data


def partition(data, k):
    """Breaks data up into k folds"""
    return np.array_split(data, k)


def train_validation_split(folds, i):
    """ Remove and return the ith fold from folds for validation"""
    # Copy folds so we don't affect the caller
    f = folds[:]
    validation = f.pop(i)
    return np.concatenate(f), validation


def my_accuracy_score(y1, y2):
    """ Return success rate between two label arrays"""

    if len(y1.shape) == 1:
        y1 = y1.reshape(len(y1), 1)

    if len(y2.shape) == 1:
        y2 = y2.reshape(len(y2), 1)

    return np.sum(y1 == y2) / float(y1.shape[0])


def train_model(method, train, validation):
    """Train a model on the given algorithm

    Trains a model given an sklearn algorithm and returns success rate
    on the validation set

    """
    train_y = train[:, -1:]
    train_X = train[:, :-1]
    validation_y = validation[:, -1:]
    validation_X = validation[:, :-1]
    method.fit(train_X, train_y.ravel())
    ypred = method.predict(validation_X)
    return my_accuracy_score(validation_y, ypred)


def my_cross_val(method, X, y, k):
    """Perform k-fold cross-validation

    Parameters
    ----------
    method : object implmenting fit
        The object used to fit a model to the data
    X : array-like
        The data to be fitted
    y : array-like
        The target data to predict
    k : int
        Number of folds in CV

    Returns
    -------
    scores: list
        Success rates after running cv
    """

    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)

    scores = []
    data = randomize(X, y)
    folds = partition(data, k)

    for i in range(len(folds)):
        train, validation = train_validation_split(folds, i)
        acc = train_model(method, train, validation)
        scores.append(acc)
    return scores
