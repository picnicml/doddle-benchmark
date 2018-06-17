# -*- coding: utf-8 -*-
""" Logistic Regression
"""
import logging
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data into memory
file_name = "/home/dan/Desktop/picnicml/doddle-benchmark/src/main/resources/Logistic_data.csv"
data = pd.read_csv(file_name, header=None).values
training_set_size = 80000
benchmark_iterations = 100

# Split the data
X_tr, y_tr = data[:training_set_size, :-1], data[:training_set_size, -1]
X_te, y_te = data[training_set_size:, :-1], data[training_set_size:, -1]


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        execution_time = time.time() - ts
        logging.info('Method: {} - {} seconds'.format(method.__name__,
                                                      execution_time))
        return execution_time, result

    return timed


@timeit
def run_fit(model):
    return model.fit(X_tr, y_tr)


@timeit
def run_prediction(model):
    return model.predict(X_te)


models = [LogisticRegression(tol=1e-4, C=1e10, solver='lbfgs')
          for _ in range(benchmark_iterations)]

training_ops = [run_fit(each_model) for each_model in models]
prediction_ops = [run_prediction(model_time[1]) for model_time in training_ops]

training_times = np.array([items[0] for items in training_ops])
y_te_pred = prediction_ops[-1][1]
prediction_times = np.array([items[0] for items in prediction_ops])

print("Training time: {:.3f}s (+/- {:.3f}s)".format(training_times.mean(), 2 * training_times.std()))
print("Prediction time: {:.3f}s (+/- {:.3f}s)".format(prediction_times.mean(), 2 * prediction_times.std()))
print("Test set Accuracy: {:.4f}".format(accuracy_score(y_te, y_te_pred)))
