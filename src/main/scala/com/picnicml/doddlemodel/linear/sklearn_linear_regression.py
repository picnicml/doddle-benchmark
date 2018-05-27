import math
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('bng_pw_linear_shuffled.csv', header=None).values

X_tr, y_tr = data[:150000, :-1], data[:150000, -1]
X_te, y_te = data[150000:, :-1], data[150000:, -1]

training_times = []
prediction_times = []

for _ in range(100):
    lin_reg = LinearRegression(normalize=False)
    t0 = time.time()
    lin_reg.fit(X_tr, y_tr)
    training_times.append(time.time() - t0)

    t0 = time.time()
    lin_reg.predict(X_te)
    prediction_times.append(time.time() - t0)

lin_reg = LinearRegression(normalize=False)
lin_reg.fit(X_tr, y_tr)
y_te_pred = lin_reg.predict(X_te)

training_times = np.array(training_times)
prediction_times = np.array(prediction_times)

print("Training time: {:.3f}s (+/- {:.3f}s)".format(training_times.mean(), 2 * training_times.std()))
print("Prediction time: {:.3f}s (+/- {:.3f}s)".format(prediction_times.mean(), 2 * prediction_times.std()))
print("Test set rmse: {:.4f}".format(math.sqrt(mean_squared_error(y_te, y_te_pred))))
