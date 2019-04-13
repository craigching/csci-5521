from sklearn.linear_model import LogisticRegression
from MyLogisticReg2 import MyLogisticReg2
from my_cross_val import my_cross_val
from datasets import (
    prepare_boston50,
    prepare_boston75,
    )
from utils import (
    report,
    wrapper_args
    )
import sys


def q3(argv=None):

    dataset, method_name, k, latex = wrapper_args(
        argv, 'q3',
        ['Boston50', 'Boston75'],
        ['MyLogisticReg2', 'LogisticRegression']
    )

    Boston50_X, Boston50_y = prepare_boston50()
    Boston75_X, Boston75_y = prepare_boston75()

    default_order = [
        ('MyLogisticReg2', 'Boston50'),
        ('MyLogisticReg2', 'Boston75'),
        ('LogisticRegression', 'Boston50'),
        ('LogisticRegression', 'Boston75')
    ]
    methods = {
        ('MyLogisticReg2', 'Boston50'):
        (MyLogisticReg2(verbose=False), Boston50_X, Boston50_y),
        ('MyLogisticReg2', 'Boston75'):
        (MyLogisticReg2(verbose=False), Boston75_X, Boston75_y),
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
