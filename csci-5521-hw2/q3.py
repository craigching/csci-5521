from sklearn.linear_model import LogisticRegression
from MultiGaussClassify import MultiGaussClassify
from my_cross_val import my_cross_val
from datasets import (
    prepare_boston50,
    prepare_boston75,
    prepare_digits
    )
from utils import (
    report,
    wrapper_args
    )
import sys


def q3(argv=None):

    dataset, method_name, k, latex = wrapper_args(
        argv, 'q3', ['Boston50', 'Boston75', 'Digits'])

    Boston50_X, Boston50_y = prepare_boston50()
    Boston75_X, Boston75_y = prepare_boston75()
    # Note that prepare_digits adds gaussian noise to the data to
    # avoid singlar covariance matrices.  For details, see
    # datasets.prepare_digits
    Digits_X, Digits_y = prepare_digits()

    default_order = [
        ('MultiGaussClassify', 'Boston50'),
        ('MultiGaussClassify', 'Boston75'),
        ('MultiGaussClassify', 'Digits'),
        ('LogisticRegression', 'Boston50'),
        ('LogisticRegression', 'Boston75'),
        ('LogisticRegression', 'Digits')
    ]

    methods = {
        ('MultiGaussClassify', 'Boston50'):
        (MultiGaussClassify(), Boston50_X, Boston50_y),
        ('MultiGaussClassify', 'Boston75'):
        (MultiGaussClassify(), Boston75_X, Boston75_y),
        ('MultiGaussClassify', 'Digits'):
        (MultiGaussClassify(linear=False), Digits_X, Digits_y),
        ('LogisticRegression', 'Boston50'):
        (LogisticRegression(), Boston50_X, Boston50_y),
        ('LogisticRegression', 'Boston75'):
        (LogisticRegression(), Boston75_X, Boston75_y),
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
    q3(sys.argv[1:])
