import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(0)

X, y = make_classification(n_samples=100000, n_features=250, n_informative=150, n_classes=2, flip_y=0.05)

X_tr, y_tr = X[:80000, :], y[:80000]
X_te, y_te = X[80000:, :], y[80000:]

# data = np.hstack((X, y.reshape(-1, 1)))
# with open('log_reg.csv', 'wb') as f:
#     np.savetxt(f, data, delimiter=',')

training_times = []
prediction_times = []

for _ in range(100):
    log_reg = LogisticRegression(tol=1e-4, C=1e10, solver='lbfgs')
    t0 = time.time()
    log_reg.fit(X_tr, y_tr)
    training_times.append(time.time() - t0)

    t0 = time.time()
    log_reg.predict(X_te)
    prediction_times.append(time.time() - t0)

log_reg = LogisticRegression(tol=1e-4, C=1e10, solver='lbfgs')
log_reg.fit(X_tr, y_tr)
y_te_pred = log_reg.predict(X_te)

training_times = np.array(training_times)
prediction_times = np.array(prediction_times)

print("Training time: {:.3f}s (+/- {:.3f}s)".format(training_times.mean(), 2 * training_times.std()))
print("Prediction time: {:.3f}s (+/- {:.3f}s)".format(prediction_times.mean(), 2 * prediction_times.std()))
print("Test set accuracy: {:.4f}".format(accuracy_score(y_te, y_te_pred)))
