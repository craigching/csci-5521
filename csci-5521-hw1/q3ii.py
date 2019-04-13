from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from my_train_test import my_train_test
from utils import (
    report,
    wrapper_args
    )

from datasets import (
    prepare_boston50,
    prepare_boston75,
    prepare_digits
    )

import sys


def q3ii(argv=None):

    dataset, method_name, k, pi, latex = wrapper_args(
        argv, 'q3ii', ['Boston50', 'Boston75', 'Digits'], include_pi=True)

    Boston50_X, Boston50_y = prepare_boston50()
    Boston75_X, Boston75_y = prepare_boston75()
    Digits_X, Digits_y = prepare_digits()

    default_order = [
        ('LinearSVC', 'Boston50'),
        ('LinearSVC', 'Boston75'),
        ('LinearSVC', 'Digits'),
        ('SVC', 'Boston50'),
        ('SVC', 'Boston75'),
        ('SVC', 'Digits'),
        ('LogisticRegression', 'Boston50'),
        ('LogisticRegression', 'Boston75'),
        ('LogisticRegression', 'Digits')
    ]

    methods = {('LinearSVC', 'Boston50'):
               (LinearSVC(), Boston50_X, Boston50_y),
               ('LinearSVC', 'Boston75'):
               (LinearSVC(), Boston75_X, Boston75_y),
               ('LinearSVC', 'Digits'):
               (LinearSVC(), Digits_X, Digits_y),
               ('SVC', 'Boston50'):
               (SVC(), Boston50_X, Boston50_y),
               ('SVC', 'Boston75'):
               (SVC(), Boston75_X, Boston75_y),
               ('SVC', 'Digits'):
               (SVC(), Digits_X, Digits_y),
               ('LogisticRegression', 'Boston50'):
               (LogisticRegression(), Boston50_X, Boston50_y),
               ('LogisticRegression', 'Boston75'):
               (LogisticRegression(), Boston75_X, Boston75_y),
               ('LogisticRegression', 'Digits'):
               (LogisticRegression(), Digits_X, Digits_y)}

    if dataset == 'all':
        order = default_order
    else:
        order = [(method_name, dataset)]

    for key in order:
        name, dataset = key
        method, X, y = methods[key]
        print('==============')
        print('method: {}, dataset: {}'.format(key[0], key[1]))
        scores = my_train_test(method, X, y, 0.75, k)
        report(name, dataset, scores, latex=latex)


if __name__ == '__main__':
    q3ii(sys.argv[1:])
