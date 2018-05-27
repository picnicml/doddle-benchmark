import time

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(0)

mnist = fetch_mldata('MNIST original')
data = np.hstack((mnist.data, mnist.target.reshape(-1, 1)))
np.random.shuffle(data)

# with open('mnist.csv', 'wb') as f:
#     np.savetxt(f, data.astype(int), fmt='%i', delimiter=',')

X_tr, y_tr = data[:60000, :-1], data[:60000, -1]
X_te, y_te = data[60000:, :-1], data[60000:, -1]
X_tr /= 255
X_te /= 255

training_times = []
prediction_times = []

for _ in range(2):
    softmax = LogisticRegression(tol=1e-4, C=1e10, solver='lbfgs', multi_class='multinomial')
    t0 = time.time()
    softmax.fit(X_tr, y_tr)
    training_times.append(time.time() - t0)

    t0 = time.time()
    softmax.predict(X_te)
    prediction_times.append(time.time() - t0)

softmax = LogisticRegression(tol=1e-4, C=1e10, solver='lbfgs', multi_class='multinomial')
softmax.fit(X_tr, y_tr)
y_te_pred = softmax.predict(X_te)

training_times = np.array(training_times)
prediction_times = np.array(prediction_times)

print("Training time: {:.3f}s (+/- {:.3f}s)".format(training_times.mean(), 2 * training_times.std()))
print("Prediction time: {:.3f}s (+/- {:.3f}s)".format(prediction_times.mean(), 2 * prediction_times.std()))
print("Test set accuracy: {:.4f}".format(accuracy_score(y_te, y_te_pred)))
