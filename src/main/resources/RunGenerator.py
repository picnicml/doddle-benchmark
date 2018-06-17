#!/usr/bin/python
""" Python Script to easily generate data for use in benchmarking

    Change the parameters and comment/uncomment below for the given dataset you would
    like to create.

    Available:
        Regression
        Classification

"""
import numpy as np
from sklearn.datasets import make_classification, make_regression

MAKE_CLASSIFICATION = True
FILE_NAME = "{0}.csv".format("Regression_data")
SAMPLE_SIZE = 100000


def run_classification():
    return make_classification(n_samples=SAMPLE_SIZE, n_features=20, n_informative=2,
                               n_redundant=2, n_repeated=0, n_classes=2,
                               n_clusters_per_class=2, weights=None, flip_y=0.01,
                               class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                               shuffle=True, random_state=None)


def run_regression():
    return make_regression(n_samples=SAMPLE_SIZE, n_features=30, n_informative=10,
                           n_targets=1, bias=0.0, effective_rank=None,
                           tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                           random_state=None)


if __name__ == '__main__':
    # Generate the data given the settings defined above
    (X, y) = run_classification() if MAKE_CLASSIFICATION else run_regression()
    # Reshape if required
    y = y if len(y.shape) != 1 else y.reshape(SAMPLE_SIZE, 1)
    # Concatenate the Arrays together and write to .csv
    np.savetxt(FILE_NAME, np.hstack((X, y)), delimiter=",")
