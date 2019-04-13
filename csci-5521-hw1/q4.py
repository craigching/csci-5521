from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from datasets import prepare_digits
from my_cross_val import my_cross_val
from proj import (
    rand_proj,
    quad_proj
    )
from utils import (
    report,
    wrapper_args
    )

import sys


def q4(argv):

    dataset, method_name, k, latex = wrapper_args(
        argv, 'q4', ['X1', 'X2', 'X3'])

    Digits_X, Digits_y = prepare_digits()
    X1 = rand_proj(Digits_X, 32)
    X2 = quad_proj(Digits_X)
    X3 = rand_proj(X2, 64)

    default_order = [
        ('LinearSVC', 'X1'),
        ('LinearSVC', 'X2'),
        ('LinearSVC', 'X3'),
        ('SVC', 'X1'),
        ('SVC', 'X2'),
        ('SVC', 'X3'),
        ('LogisticRegression', 'X1'),
        ('LogisticRegression', 'X2'),
        ('LogisticRegression', 'X3')
    ]

    methods = {('LinearSVC', 'X1'):
               (LinearSVC(), X1, Digits_y),
               ('LinearSVC', 'X2'):
               (LinearSVC(), X2, Digits_y),
               ('LinearSVC', 'X3'):
               (LinearSVC(), X3, Digits_y),
               ('SVC', 'X1'):
               (SVC(), X1, Digits_y),
               ('SVC', 'X2'):
               (SVC(), X2, Digits_y),
               ('SVC', 'X3'):
               (SVC(), X3, Digits_y),
               ('LogisticRegression', 'X1'):
               (LogisticRegression(), X1, Digits_y),
               ('LogisticRegression', 'X2'):
               (LogisticRegression(), X2, Digits_y),
               ('LogisticRegression', 'X3'):
               (LogisticRegression(), X3, Digits_y)}

    if dataset == 'all':
        order = default_order
    else:
        order = [(method_name, dataset)]

    for key in order:
        name, dataset = key
        method, X, y = methods[key]
        print('==============')
        print('method: {}, dataset: {}'.format(name, dataset))
        scores = my_cross_val(method, X, y, k)
        report(name, dataset, scores, latex=latex)


if __name__ == '__main__':
    q4(sys.argv[1:])
