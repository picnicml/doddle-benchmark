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

softmax = LogisticRegression(tol=1e-4, C=1e6, solver='lbfgs', multi_class='multinomial')

t0 = time.time()
softmax.fit(X_tr, y_tr)
run_time = time.time() - t0
print("Training time: {:.3f}s".format(run_time))

t0 = time.time()
y_te_pred = softmax.predict(X_te)
run_time = time.time() - t0
print("Prediction time: {:.3f}s".format(run_time))
print("Test set accuracy: {:.4f}".format(accuracy_score(y_te, y_te_pred)))

# Training time: 20.357s
# Prediction time: 0.070s
# Test set accuracy: 0.9235
