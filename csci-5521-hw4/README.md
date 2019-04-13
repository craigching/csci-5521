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
python q3.py -d <Boston50|Boston75> -m <MyFLDA2|LogisticRegression> -k [num folds] -l [latex output on]
```

Note that for this assignment, `k=5` is the default.

## Module Description

### hw4-cching.pdf

The pdf documenation required for this assignment.  Contains solutions
to questions 1, 2, extra credit and output of running `q3.py` and
`q4.py`

### hw4-cching.tex

The source for hw3-cching.pdf.

### MyFLDA2.py

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

This section provides detailed examples of output for the two wrapper
scripts

### q3 Output
```
$ python q3.py
Finding optimal threshold for MyFLDA2 on Boston50 ...
iteration: 0
iteration: 100
iteration: 200
iteration: 300
iteration: 400
iteration: 500
Finding optimal threshold for MyFLDA2 on Boston75 ...
iteration: 0
iteration: 100
iteration: 200
iteration: 300
iteration: 400
iteration: 500
Done.
==============
method: MyFLDA2, dataset: Boston50
name: MyFLDA2
dataset: Boston50
error rates:
0.1373
0.1782
0.0792
0.1485
0.1881
mean: 0.1463
std dev: 0.0384
==============
method: MyFLDA2, dataset: Boston75
name: MyFLDA2
dataset: Boston75
error rates:
0.1078
0.1089
0.0891
0.1287
0.1089
mean: 0.1087
std dev: 0.0125
==============
method: LogisticRegression, dataset: Boston50
name: LogisticRegression
dataset: Boston50
error rates:
0.1275
0.1584
0.1089
0.1386
0.1386
mean: 0.1344
std dev: 0.0162
==============
method: LogisticRegression, dataset: Boston75
name: LogisticRegression
dataset: Boston75
error rates:
0.0784
0.0990
0.1089
0.1188
0.1188
mean: 0.1048
std dev: 0.0151
```
