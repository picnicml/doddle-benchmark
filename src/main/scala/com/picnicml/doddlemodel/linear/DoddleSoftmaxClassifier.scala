package com.picnicml.doddlemodel.linear

import com.picnicml.doddlemodel.TimingUtils
import com.picnicml.doddlemodel.data.loadCsvDataset
import com.picnicml.doddlemodel.metrics.accuracy

object DoddleSoftmaxClassifier extends App with TimingUtils {
  val data = loadCsvDataset("src/main/scala/com/picnicml/doddlemodel/linear/mnist.csv", headerLine = false)
  val (xTr, yTr) = (data(0 until 60000, 0 to -2) / 255.0, data(0 until 60000, -1))
  val (xTe, yTe) = (data(60000 to -1, 0 to -2) / 255.0, data(60000 to -1, -1))

  val softmax = SoftmaxClassifier(2e-4)
  val trainedSoftmax = time("Training") { softmax.fit(xTr, yTr) }
  val yTePred = time("Prediction") { trainedSoftmax.predict(xTe) }

  println(s"Test set accuracy: ${accuracy(yTe, yTePred)}")
}

// Training time: 20.786s
// Prediction time: 0.09s
// Test set accuracy: 0.9209
