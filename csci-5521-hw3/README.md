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
% python q3.py -h
python q3.py -d <Boston50|Boston75> -m <MyLogisticReg2|LogisticRegression> -k [num folds] -l [latex output on]
```

Note that for this assignment, `k=5` is the default.

### q4

For q4, simply run `python q4.py`.  By default, all combinations
asked for in the assignment will be run.  Before each algorithm and
data set combination, the information about what is being run is
printed to the console.

If desired, you can specify the following command line parameters:

```
% python q4.py -h
python q4.py -d <Digits> -m <MyLogisticRegGen|LogisticRegression> -k [num folds] -l [latex output on]
```

Note that for this assignment, `k=5` is the default.

## Module Description

### hw3-cching.pdf

The pdf documenation required for this assignment.  Contains solutions
to questions 1 and 2 and output of running `q3.py` and `q4.py`

### hw3-cching.tex

The source for hw3-cching.pdf.

### MyLogisticReg2.py

Implements two-class logistic regression using the `sigmoid`
function. This class just sets parameters that work specifically for
the boston housing data sets, such as the learning rate `eta`, the
number of `iterations` of gradient descent, and a `reg` parameter for
regularization. It also provides a `sigmoid` function and the gradient
and loss functions specific to two-class logistic regression.  The
parameters are documented here as well:

```
  reg = 1.0
  iterations = 500
  eta = 0.01
```

### MyLogisticRegGen.py

Implements multi-class logistic regression using the `softmax`
function. This class just sets parameters that work specifically for
the digits data set, such as the learning rate `eta`, the number of
`iterations` of gradient descent, and a `reg` parameter for
regularization. It also provides a `softmax` function and the gradient
and loss function specific to multi-class logistic regression.  The
parameters are documented here as well:

```
  reg = 1.0
  iterations = 1000
  eta = 0.00001
```

### BaseLogisticRegression.py

A base implementation of logistic regression that handles tasks common
to all logistic regression implementations including scaling the data,
fitting an intercept, and transforming `y` to one-hot for multi-class
logistic regression.

### optimize.py

Contains the implemenation of Batch Gradient Descent used by both
implementations of logistic regression, `MyLogisticReg2` and
`MyLogisticRegGen`.

### preprocessing.py

Provides the common preprocessing functions `my_label_binarize` to
one-hot encode the response variables, `zscale` to standardize the
features, and `fit_intercept` to add an intercept term to the dataset.

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

### q4.py

Wrapper script, see the section on `q4` to run.

## Output Examples

This section provides detailed examples of output for the two wrapper
scripts

### q3 Output
```
% python q3.py
==============
method: MyLogisticReg2, dataset: Boston50
name: MyLogisticReg2
dataset: Boston50
error rates:
0.0882
0.1287
0.1287
0.1287
0.2079
mean: 0.1365
std dev: 0.0390
==============
method: MyLogisticReg2, dataset: Boston75
name: MyLogisticReg2
dataset: Boston75
error rates:
0.1275
0.0792
0.0594
0.1386
0.0990
mean: 0.1007
std dev: 0.0294
==============
method: LogisticRegression, dataset: Boston50
name: LogisticRegression
dataset: Boston50
error rates:
0.1176
0.0792
0.1584
0.1584
0.1584
mean: 0.1344
std dev: 0.0318
==============
method: LogisticRegression, dataset: Boston75
name: LogisticRegression
dataset: Boston75
error rates:
0.1176
0.0990
0.1188
0.0891
0.0990
mean: 0.1047
std dev: 0.0116
```

### q4 Output

```
% python q4.py
==============
method: MyLogisticRegGen, dataset: Digits
name: MyLogisticRegGen
dataset: Digits
error rates:
0.0333
0.0278
0.0390
0.0418
0.0362
mean: 0.0356
std dev: 0.0048
==============
method: LogisticRegression, dataset: Digits
name: LogisticRegression
dataset: Digits
error rates:
0.0417
0.0417
0.0362
0.0418
0.0334
mean: 0.0390
std dev: 0.0035
```
