from sklearn.linear_model import LogisticRegression
from MyLogisticRegGen import MyLogisticRegGen
from my_cross_val import my_cross_val
from datasets import prepare_digits
from utils import (
    report,
    wrapper_args
    )
import sys


def q4(argv=None):

    dataset, method_name, k, latex = wrapper_args(
        argv, 'q4',
        ['Digits'],
        ['MyLogisticRegGen', 'LogisticRegression'])

    Digits_X, Digits_y = prepare_digits(want_noise=False)

    default_order = [
        ('MyLogisticRegGen', 'Digits'),
        ('LogisticRegression', 'Digits')
    ]
    methods = {
        ('MyLogisticRegGen', 'Digits'):
        (MyLogisticRegGen(verbose=False), Digits_X, Digits_y),
        ('LogisticRegression', 'Digits'):
        (LogisticRegression(), Digits_X, Digits_y)
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
    q4(sys.argv[1:])
