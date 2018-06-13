#!/usr/bin/python
""" Command Line Loader of Scikit-Learning Datasets
"""
import argparse

from sklearn.datasets.base import *
from sklearn.datasets.california_housing import *


def get_california_data():
    all_data = fetch_california_housing()
    return all_data.data, all_data.target


def validate_selection(name):
    if name in DATA.keys():
        print("About to move data for '{0}' to the resources folder.".format(name))
    else:
        print("The selection you've made, '{0}', is not available".format(name))
        sys.exit()


def data_fetcher(name):
    return DATA.get(name)


def write_data_to_csv(name, X, y):
    # Reshape if required
    y = y if len(y.shape) != 1 else y.reshape(y.shape[0], 1)
    # Concatenate the Arrays together and write to .csv
    np.savetxt("{0}.csv".format(name), np.hstack((X, y)), delimiter=",")


def dataset_selection(name):
    updated_name = name.lower()
    validate_selection(updated_name)
    (X, y) = data_fetcher(updated_name)
    write_data_to_csv(updated_name, X, y)


DATA = dict(boston=load_boston(return_X_y=True),
            breast_cancer=load_breast_cancer(return_X_y=True),
            california_housing=get_california_data(),
            diabetes=load_diabetes(return_X_y=True),
            digits=load_digits(return_X_y=True),
            iris=load_iris(return_X_y=True),
            linnerud=load_linnerud(return_X_y=True),
            wine=load_wine(return_X_y=True))

if __name__ == '__main__':
    # Use 'argparse' to control user inputs
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Name of the dataset to move to the resources folder.\n{0}'.format(list(DATA.keys())))
    # Parse the Arguments from the Command Line
    args = parser.parse_args()
    # Validate the input, download the dataset and move to the resources folder
    dataset_selection(args.dataset)
