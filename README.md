## doddle-benchmark
Benchmarking [doddle-model](https://github.com/picnicml/doddle-model) implementations.

All experiments are run multiple times (iterations) for all implementations and with fixed hyperparameters selected in a way such that models yielded similar test set performance.

#### Softmax Classifier
- MNIST dataset with 60000 training examples and 10000 test examples (784 features)
- each experiment ran for 50 iterations

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

For details see [scikit-learn code](src/main/scala/com/picnicml/doddlemodel/linear/sklearn_softmax_classifier.py) and [doddle-model code](src/main/scala/com/picnicml/doddlemodel/linear/DoddleSoftmaxClassifier.scala).
