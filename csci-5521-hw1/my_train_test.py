from my_cross_val import (
    randomize,
    train_model
    )
from math import ceil


def my_split_train_test(X, y, pi):
    data = randomize(X, y)
    idx = int(ceil(pi * len(data)))
    return data[:idx, :], data[idx:, :]


def my_train_test(method, X, y, pi, k):

    if len(y.shape) == 1:
        y = y.reshape(len(y), 1)

    scores = []
    for i in range(k):
        train, validation = my_split_train_test(X, y, pi)
        acc = train_model(method, train, validation)
        scores.append(acc)
    return scores
