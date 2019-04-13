from __future__ import print_function

import numpy as np
import sys
import getopt


def eprint(*args, **kwargs):
    """Output to stderr"""
    print(*args, file=sys.stderr, **kwargs)


def out_latex(name, dataset, errors):
    """Latex text output of info from running tests"""

    tabular = []
    header = []
    err = []
    for i, error in enumerate(errors, 1):
        tabular.append(' | l')
        header.append('Fold {} & '.format(i))
        err.append('{:.4f} & '.format(error))

    header.append('mean &')
    header.append(' std dev')

    err.append('{:.4f} &'.format(np.mean(errors)))
    err.append(' {:.4f}'.format(np.std(errors)))

    tabular.append(' | l | l |')

    eprint('\\noindent method: {}'.format(name))
    eprint('')
    eprint('\\noindent dataset: {}'.format(dataset))
    eprint('')
    eprint('\\begin{center}')
    eprint('\t\\begin{{tabular}}  {{{} }}'.format(''.join(tabular)))
    eprint('\t\\hline')
    eprint('\t{}\\\\ \hline'.format(''.join(header)))
    eprint('\t{}\\\\'.format(''.join(err)))
    eprint('\t\\hline')
    eprint('\t\\end{tabular}')
    eprint('\\end{center}')


def out_plain(name, dataset, errors):
    """Plain text output of info from running tests"""
    print('name: {}'.format(name))
    print('dataset: {}'.format(dataset))
    print('error rates:')
    for error in errors:
        print('{:.4f}'.format(error))
    print('mean: {:.4f}'.format(np.mean(errors)))
    print('std dev: {:.4f}'.format(np.std(errors)))


def report(name, dataset, scores, latex=False):
    """Output error rates and other info, dispatches to plain or latex"""

    errors = np.array(scores)
    errors = 1 - errors
    if latex:
        out_latex(name, dataset, errors)
    else:
        out_plain(name, dataset, errors)


def wrapper_help(name, datasets, include_pi=False):
    """Display help including command-line options"""

    piopt = ''

    if include_pi:
        piopt = ' -p <pct split>'

    print(('python ' + name + '.py'
           + ' -d <{}>'.format('|'.join(datasets)) +
           ' -m <LinearSVC|SVC|LogisticRegression>'
           ' -k [num folds]'
           + piopt +
           ' -l [latex output on]'))


def wrapper_args(argv, name, datasets, include_pi=False):
    """ Handles command-line arguments"""
    dataset = 'all'
    method_name = None
    k = 10
    pi = 0.75
    latex_output = False

    if argv:
        try:
            opts, args = getopt.getopt(argv, 'hlm:d:k:', [])
        except getopt.GetOptError:
            wrapper_help(name, datasets, include_pi)
            sys.exit()

        for opt, arg in opts:
            if opt == '-h':
                wrapper_help(name, datasets, include_pi)
                sys.exit(2)
            elif opt == '-l':
                latex_output = True
            elif opt == '-m':
                method_name = arg
            elif opt == '-d':
                dataset = arg
            elif opt == '-k':
                k = int(arg)
            elif opt == '-p':
                pi = int(arg)

    if include_pi:
        return (dataset, method_name, k, pi, latex_output)
    else:
        return (dataset, method_name, k, latex_output)
