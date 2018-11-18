package com.picnicml.doddlemodel.linear

import breeze.linalg.DenseVector
import breeze.stats.{mean, stddev}
import com.picnicml.doddlemodel.TimingUtils
import com.picnicml.doddlemodel.data.loadCsvDataset
import com.picnicml.doddlemodel.metrics.accuracy
import com.picnicml.doddlemodel.syntax.ClassifierSyntax._

object DoddleSoftmaxClassifier extends App with TimingUtils {
  val data = loadCsvDataset("src/main/scala/com/picnicml/doddlemodel/linear/mnist.csv", headerLine = false)
  val (xTr, yTr) = (data(0 until 60000, 0 to -2) / 255.0, data(0 until 60000, -1))
  val (xTe, yTe) = (data(60000 to -1, 0 to -2) / 255.0, data(60000 to -1, -1))

  val softmax = SoftmaxClassifier(1e-4)

  def trainingPredictionTimes: (Double, Double) = {
    val (trainedSoftmax, timeTr) = time("Training") { softmax.fit(xTr, yTr) }
    val (_, timePr) = time("Prediction") { trainedSoftmax.predict(xTe) }
    (timeTr, timePr)
  }

  val (timesTrTot, timesPrTot) = (1 to 50).foldLeft((List[Double](), List[Double]())) { case ((timesTr, timesPr), _) =>
    val (timeTr, timePr) = trainingPredictionTimes
    (timeTr :: timesTr, timePr :: timesPr)
  }

  val trainedSoftmax = softmax.fit(xTr, yTr)
  val yTePred = trainedSoftmax.predict(xTe)

  val timesTr = DenseVector(timesTrTot.toArray)
  val timesPr = DenseVector(timesPrTot.toArray)

  println(f"Training time: ${mean(timesTr)}%1.3fs (+/- ${2 * stddev(timesTr)}%1.3fs)")
  println(f"Prediction time: ${mean(timesPr)}%1.3fs (+/- ${2 * stddev(timesPr)}%1.3fs)")
  println(f"Test set accuracy: ${accuracy(yTe, yTePred)}%1.4f")
}
