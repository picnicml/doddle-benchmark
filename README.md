## doddle-benchmark
Benchmarking [doddle-model](https://github.com/picnicml/doddle-model) implementations.

All experiments ran multiple times (iterations) for all implementations and with fixed hyperparameters, selected in a way such that models yielded similar test set performance.

#### Linear Regression
- dataset with 150000 training examples and 27147 test examples (10 features)
- each experiment ran for 100 iterations
- [scikit-learn code](src/main/scala/com/picnicml/doddlemodel/linear/sklearn_linear_regression.py), [doddle-model code](src/main/scala/com/picnicml/doddlemodel/linear/DoddleLinearRegression.scala)

<table>
<tr>
  <th>Implementation</th>
  <th>RMSE</th>
  <th>Training Time</th>
  <th>Prediction Time</th>
</tr>
<tr>
  <td>scikit-learn</td>
  <td>3.0936</td>
  <td>0.042s (+/- 0.014s)</td>
  <td>0.002s (+/- 0.002s)</td>
</tr>
<tr>
  <td>doddle-model</td>
  <td>3.0936</td>
  <td>0.053s (+/- 0.061s)</td>
  <td>0.002s (+/- 0.004s)</td>
</tr>
</table>

#### Logistic Regression
- dataset with 80000 training examples and 20000 test examples (250 features)
- each experiment ran for 100 iterations
- [scikit-learn code](src/main/scala/com/picnicml/doddlemodel/linear/sklearn_logistic_regression.py), [doddle-model code](src/main/scala/com/picnicml/doddlemodel/linear/DoddleLogisticRegression.scala)

<table>
<tr>
  <th>Implementation</th>
  <th>Accuracy</th>
  <th>Training Time</th>
  <th>Prediction Time</th>
</tr>
<tr>
  <td>scikit-learn</td>
  <td>0.8389</td>
  <td>2.789s (+/- 0.090s)</td>
  <td>0.005s (+/- 0.006s)</td>
</tr>
<tr>
  <td>doddle-model</td>
  <td>0.8377</td>
  <td>3.080s (+/- 0.665s)</td>
  <td>0.025s (+/- 0.025s)</td>
</tr>
</table>

#### Softmax Classifier
- MNIST dataset with 60000 training examples and 10000 test examples (784 features)
- each experiment ran for 50 iterations
- [scikit-learn code](src/main/scala/com/picnicml/doddlemodel/linear/sklearn_softmax_classifier.py), [doddle-model code](src/main/scala/com/picnicml/doddlemodel/linear/DoddleSoftmaxClassifier.scala)

<table>
<tr>
  <th>Implementation</th>
  <th>Accuracy</th>
  <th>Training Time</th>
  <th>Prediction Time</th>
</tr>
<tr>
  <td>scikit-learn</td>
  <td>0.9234</td>
  <td>21.243s (+/- 0.303s)</td>
  <td>0.074s (+/- 0.018s)</td>
</tr>
<tr>
  <td>doddle-model</td>
  <td>0.9223</td>
  <td>25.749s (+/- 1.813s)</td>
  <td>0.042s (+/- 0.032s)</td>
</tr>
</table>
