from sklearn.linear_model import LogisticRegression
from my_cross_val import my_cross_val
from datasets import prepare_boston50, prepare_boston75
from utils import (
    report,
    wrapper_args
    )
from MyFLDA2 import MyFLDA2
import sys
import numpy as np


def find_best_myflda2(Method, X, y, k):
    # Find the optimal threshold for MyFLDA2 for the given dataset
    threshold_perf = []
    for i in range(X.shape[0]):
        if i % 100 == 0:
            print('iteration: {}'.format(i))
        x = X[i]
        lda = Method(threshold=x)
        scores = my_cross_val(lda, X, y, k)
        threshold_perf.append((np.mean(scores), lda))

    # Include using the average of the class means as the threshold
    lda = Method()
    scores = my_cross_val(lda, X, y, k)
    threshold_perf.append((np.mean(scores), lda))

    # Sort them from highest to lowest, my_cross_val returns
    # accuracies, and return the best performing model
    threshold_perf.sort(key=lambda p: p[0], reverse=True)
    return threshold_perf[0][1]


def q3(argv=None):

    dataset, method_name, k, latex = wrapper_args(
        argv,
        'q3',
        ['Boston50', 'Boston75'],
        ['MyFLDA2', 'LogisticRegression']
    )

    Boston50_X, Boston50_y = prepare_boston50()
    Boston75_X, Boston75_y = prepare_boston75()

    default_order = [
        ('MyFLDA2', 'Boston50'),
        ('MyFLDA2', 'Boston75'),
        ('LogisticRegression', 'Boston50'),
        ('LogisticRegression', 'Boston75')
    ]

    # Find the optimal separation for the training set
    print('Finding optimal threshold for MyFLDA2 on Boston50 ...')
    myflda_boston50 = find_best_myflda2(MyFLDA2, Boston50_X, Boston50_y, k)

    print('Finding optimal threshold for MyFLDA2 on Boston75 ...')
    myflda_boston75 = find_best_myflda2(MyFLDA2, Boston75_X, Boston75_y, k)

    print('Done.')

    methods = {
        ('MyFLDA2', 'Boston50'):
        (myflda_boston50, Boston50_X, Boston50_y),
        ('MyFLDA2', 'Boston75'):
        (myflda_boston75, Boston75_X, Boston75_y),
        ('LogisticRegression', 'Boston50'):
        (LogisticRegression(), Boston50_X, Boston50_y),
        ('LogisticRegression', 'Boston75'):
        (LogisticRegression(), Boston75_X, Boston75_y)
    }

    if dataset == 'all':
        order = default_order
    else:
        order = [(method_name, dataset)]

    for key in order:
        name, dataset = key
        method, X, y = methods[key]
        print('==============')
        print('method: {}, dataset: {}'.format(key[0], key[1]))
        scores = my_cross_val(method, X, y, k)
        report(name, dataset, scores, latex=latex)


if __name__ == '__main__':
    q3(sys.argv[1:])
