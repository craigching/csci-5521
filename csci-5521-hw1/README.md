Craig Ching

\#1452647

chin0007@umn.edu

## Pre-requisites

This implementation requires python 2.7 and has been tested on the lab
machine `csel-kh1260-01`:

```
chin0007@csel-kh1260-01:/home/chin0007 $ python -V
Python 2.7.12
```

## Running the assignment

This section describes how to run the programs required by the assignment

### q3i

For q3i, simply run `python q3i.py`.  By default, all combinations
asked for in the assignment will be run.  Before each algorithm and
data set combination, the information about what is being run is
printed to the console.

If desired, you can specify the following command line parameters:

```
$ python q3i.py -h
python q3i.py -d <Boston50|Boston75|Digits> -m <LinearSVC|SVC|LogisticRegression> -k [num folds] -l [latex output on]
```

### q3ii

For q3ii, simply run `python q3ii.py`.  By default, all combinations
asked for in the assignment will be run.  Before each algorithm and
data set combination, the information about what is being run is
printed to the console.

If desired, you can specify the following command line parameters:

```
$ python q3ii.py -h
python q3ii.py -d <Boston50|Boston75|Digits> -m <LinearSVC|SVC|LogisticRegression> -k [num folds] -p <pct split> -l [latex output on]
```

### q4

For q4, simply run `python q4.py`.  By default, all combinations asked
for in the assignment will be run.  Before each algorithm and data set
combination, the information about what is being run is printed to the
console.

If desired, you can specify the following command line parameters:

```
$ python q4.py -h
python q4.py -d <X1|X2|X3> -m <LinearSVC|SVC|LogisticRegression> -k [num folds] -l [latex output on]
```

## Module Description

### hw1-cching.pdf

The pdf documenation required for this assignment.

### hw2-cching.tex

The source for hw1-cching.pdf.

### datasets.py

This is a utility module that provides functions to load the data
sets.

### utils.py

This is another utility module that provides a report function that
can output either plain-text or latex.  Also provided by `utils.py` is
command-line argument handling.

### my_cross_val.py

This implements the `my_cross_val` function and supporting functions.
Note that my\_train\_test.py depends on this module.

### my_train_test.py

This implements the `my_train_test` function and supporting functions.
This module depends on `my_cross_val` for some of the shared
implementations like `randomize` and `train_model`

### proj.py

This contains implementations for `rand_proj` and `quad_proj`

### q3i.py

Wrapper script, see the section on `q3i` to run.

### q3ii.py

Wrapper script, see the section on `q3ii` to run.

### q4.py

Wrapper script, see the section on `q4` to run.

## Output Examples

This section provides detailed examples of output for the three
wrapper scripts

### q3i

```
$ python q3i.py
==============
method: LinearSVC, dataset: Boston50
name: LinearSVC
dataset: Boston50
error rates:
0.1765
0.1961
0.4314
0.1961
0.2745
0.1765
0.2600
0.3000
0.3600
0.2600
mean: 0.2631
std dev: 0.0795
==============
method: LinearSVC, dataset: Boston75
name: LinearSVC
dataset: Boston75
error rates:
0.2353
0.2157
0.0392
0.0980
0.0392
0.2157
0.1000
0.2400
0.1600
0.1600
mean: 0.1503
std dev: 0.0735
==============
method: LinearSVC, dataset: Digits
name: LinearSVC
dataset: Digits
error rates:
0.0500
0.0611
0.0833
0.0500
0.0111
0.0500
0.0389
0.0782
0.0335
0.0335
mean: 0.0490
std dev: 0.0205
==============
method: SVC, dataset: Boston50
name: SVC
dataset: Boston50
error rates:
0.3333
0.4510
0.2745
0.3529
0.3333
0.3137
0.3000
0.4000
0.2800
0.3200
mean: 0.3359
std dev: 0.0516
==============
method: SVC, dataset: Boston75
name: SVC
dataset: Boston75
error rates:
0.4118
0.2353
0.1569
0.2745
0.1765
0.2941
0.1600
0.2400
0.2600
0.2600
mean: 0.2469
std dev: 0.0716
==============
method: SVC, dataset: Digits
name: SVC
dataset: Digits
error rates:
0.4333
0.5000
0.4333
0.5722
0.5444
0.4444
0.4278
0.4358
0.4637
0.3631
mean: 0.4618
std dev: 0.0583
==============
method: LogisticRegression, dataset: Boston50
name: LogisticRegression
dataset: Boston50
error rates:
0.1765
0.1961
0.1176
0.1765
0.1176
0.0784
0.1200
0.1000
0.1000
0.1800
mean: 0.1363
std dev: 0.0396
==============
method: LogisticRegression, dataset: Boston75
name: LogisticRegression
dataset: Boston75
error rates:
0.0980
0.0980
0.0392
0.0980
0.1569
0.0784
0.1200
0.1000
0.0800
0.1000
mean: 0.0969
std dev: 0.0285
==============
method: LogisticRegression, dataset: Digits
name: LogisticRegression
dataset: Digits
error rates:
0.0056
0.0500
0.0333
0.0556
0.0389
0.0444
0.0389
0.0503
0.0559
0.0223
mean: 0.0395
std dev: 0.0150
```

### q3ii

```
$ python q3ii.py
==============
method: LinearSVC, dataset: Boston50
name: LinearSVC
dataset: Boston50
error rates:
0.1667
0.3016
0.2143
0.4603
0.2302
0.3095
0.2540
0.1825
0.4127
0.4206
mean: 0.2952
std dev: 0.0994
==============
method: LinearSVC, dataset: Boston75
name: LinearSVC
dataset: Boston75
error rates:
0.2143
0.2698
0.1587
0.1111
0.1508
0.1984
0.4206
0.1508
0.2222
0.1032
mean: 0.2000
std dev: 0.0884
==============
method: LinearSVC, dataset: Digits
name: LinearSVC
dataset: Digits
error rates:
0.0601
0.0334
0.0445
0.0713
0.0512
0.0624
0.0557
0.0490
0.0668
0.0379
mean: 0.0532
std dev: 0.0117
==============
method: SVC, dataset: Boston50
name: SVC
dataset: Boston50
error rates:
0.3730
0.3333
0.3889
0.2540
0.4206
0.3810
0.3254
0.3333
0.3651
0.3016
mean: 0.3476
std dev: 0.0457
==============
method: SVC, dataset: Boston75
name: SVC
dataset: Boston75
error rates:
0.2698
0.2698
0.2302
0.2302
0.2619
0.2698
0.2381
0.3492
0.2302
0.2698
mean: 0.2619
std dev: 0.0339
==============
method: SVC, dataset: Digits
name: SVC
dataset: Digits
error rates:
0.4967
0.5501
0.7149
0.4744
0.5367
0.6615
0.4788
0.5412
0.4677
0.5033
mean: 0.5425
std dev: 0.0787
==============
method: LogisticRegression, dataset: Boston50
name: LogisticRegression
dataset: Boston50
error rates:
0.1270
0.1270
0.1111
0.1429
0.1111
0.1349
0.1190
0.1111
0.0952
0.1429
mean: 0.1222
std dev: 0.0147
==============
method: LogisticRegression, dataset: Boston75
name: LogisticRegression
dataset: Boston75
error rates:
0.1270
0.0952
0.1032
0.1111
0.1190
0.0635
0.0556
0.0794
0.1032
0.1190
mean: 0.0976
std dev: 0.0230
==============
method: LogisticRegression, dataset: Digits
name: LogisticRegression
dataset: Digits
error rates:
0.0445
0.0312
0.0468
0.0379
0.0535
0.0334
0.0379
0.0379
0.0423
0.0401
mean: 0.0405
std dev: 0.0062
```

### q4

```
$ python q4.py
==============
method: LinearSVC, dataset: X1
name: LinearSVC
dataset: X1
error rates:
0.0889
0.1167
0.1278
0.1500
0.0889
0.0444
0.0611
0.0447
0.0503
0.0894
mean: 0.0862
std dev: 0.0348
==============
method: LinearSVC, dataset: X2
name: LinearSVC
dataset: X2
error rates:
0.0056
0.0167
0.0000
0.0222
0.0167
0.0056
0.0056
0.0168
0.0168
0.0056
mean: 0.0111
std dev: 0.0070
==============
method: LinearSVC, dataset: X3
name: LinearSVC
dataset: X3
error rates:
0.0500
0.0389
0.0556
0.0556
0.0611
0.0444
0.0667
0.0615
0.0168
0.0615
mean: 0.0512
std dev: 0.0140
==============
method: SVC, dataset: X1
name: SVC
dataset: X1
error rates:
0.9333
0.9167
0.9222
0.9278
0.9444
0.9278
0.9278
0.9609
0.9330
0.9218
mean: 0.9316
std dev: 0.0122
==============
method: SVC, dataset: X2
name: SVC
dataset: X2
error rates:
0.9278
0.9222
0.9722
0.9222
0.9167
0.9222
0.9389
0.9162
0.9330
0.8659
mean: 0.9237
std dev: 0.0248
==============
method: SVC, dataset: X3
name: SVC
dataset: X3
error rates:
0.9389
0.9222
0.9222
0.9333
0.9444
0.9167
0.9500
0.9497
0.9385
0.9218
mean: 0.9338
std dev: 0.0118
==============
method: LogisticRegression, dataset: X1
name: LogisticRegression
dataset: X1
error rates:
0.0556
0.0500
0.0444
0.0833
0.0611
0.0611
0.0889
0.0670
0.0447
0.0670
mean: 0.0623
std dev: 0.0142
==============
method: LogisticRegression, dataset: X2
name: LogisticRegression
dataset: X2
error rates:
0.0111
0.0000
0.0056
0.0222
0.0111
0.0056
0.0111
0.0112
0.0279
0.0168
mean: 0.0123
std dev: 0.0078
==============
method: LogisticRegression, dataset: X3
name: LogisticRegression
dataset: X3
error rates:
0.0333
0.0611
0.0833
0.0778
0.0556
0.0889
0.0556
0.0447
0.0782
0.0782
mean: 0.0657
std dev: 0.0174
```
