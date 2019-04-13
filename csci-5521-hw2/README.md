Craig Ching

\#1452647

chin0007@umn.edu

## Pre-requisites

This implementation requires python 2.7 and has been tested on the lab
machine `csel-kh1260-03`:

```
chin0007@csel-kh1260-03 $ python -V
Python 2.7.12
```

## Running the assignment

This section describes how to run the programs required by the assignment

### q3

For q3, simply run `python q3.py`.  By default, all combinations
asked for in the assignment will be run.  Before each algorithm and
data set combination, the information about what is being run is
printed to the console.

If desired, you can specify the following command line parameters:

```
$ python q3.py -h
python q3.py -d <Boston50|Boston75|Digits> -m <MultiGaussClassify|LogisticRegression> -k [num folds] -l [latex output on]
```

Note that for this assignment, `k=5` is the default.

## Module Description

### hw2-cching.pdf

The pdf documenation required for this assignment.  Contains solutions
to questions 1 and 2 and output of running `q3.py`

### hw2-cching.tex

The source for hw2-cching.pdf.

### MultiGaussClassify.py

This is the source for the Multivariate Gaussian Distribution
classifier.  It implements `__init__`, `fit`, and `predict` that are
compatible with typical sklearn algorithms.  Details about the
implementation can be found in comments.  At a high level, there are
implementations for both quadratic and linear discriminants, the
latter can be enabled by passing `linear=True` when constructing
`MultiGaussClassify`.

### datasets.py

This is a utility module that provides functions to load the data
sets.

### utils.py

This is another utility module that provides a report function that
can output either plain-text or latex.  Also provided by `utils.py` is
command-line argument handling.

### my_cross_val.py

This implements the `my_cross_val` function and supporting functions.

### q3.py

Wrapper script, see the section on `q3` to run.

## Output Examples

This section provides detailed examples of output for the three
wrapper scripts

### q3 Output

```
$ python q3.py
==============
method: MultiGaussClassify, dataset: Boston50
name: MultiGaussClassify
dataset: Boston50
error rates:
0.1961
0.2178
0.2475
0.1584
0.2178
mean: 0.2075
std dev: 0.0295
==============
method: MultiGaussClassify, dataset: Boston75
name: MultiGaussClassify
dataset: Boston75
error rates:
0.2451
0.2871
0.2673
0.2178
0.2772
mean: 0.2589
std dev: 0.0248
==============
method: MultiGaussClassify, dataset: Digits
name: MultiGaussClassify
dataset: Digits
error rates:
0.0667
0.0333
0.0669
0.0641
0.0362
mean: 0.0534
std dev: 0.0153
==============
method: LogisticRegression, dataset: Boston50
name: LogisticRegression
dataset: Boston50
error rates:
0.0686
0.1287
0.1584
0.1881
0.1485
mean: 0.1385
std dev: 0.0398
==============
method: LogisticRegression, dataset: Boston75
name: LogisticRegression
dataset: Boston75
error rates:
0.0980
0.0495
0.1188
0.1188
0.0693
mean: 0.0909
std dev: 0.0275
==============
method: LogisticRegression, dataset: Digits
name: LogisticRegression
dataset: Digits
error rates:
0.0333
0.0361
0.0362
0.0501
0.0334
mean: 0.0378
std dev: 0.0063
```
