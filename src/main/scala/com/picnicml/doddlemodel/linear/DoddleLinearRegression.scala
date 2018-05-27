package com.picnicml.doddlemodel.linear

import breeze.linalg.DenseVector
import breeze.stats.{mean, stddev}
import com.picnicml.doddlemodel.TimingUtils
import com.picnicml.doddlemodel.data.loadCsvDataset
import com.picnicml.doddlemodel.metrics.rmse

object DoddleLinearRegression extends App with TimingUtils {
  val data = loadCsvDataset("src/main/scala/com/picnicml/doddlemodel/linear/bng_pw_linear_shuffled.csv", headerLine = false)
  val (xTr, yTr) = (data(0 until 150000, 0 to -2), data(0 until 150000, -1))
  val (xTe, yTe) = (data(150000 to -1, 0 to -2), data(150000 to -1, -1))

  val linReg = LinearRegression()

  def trainingPredictionTimes: (Double, Double) = {
    val (trainedLinReg, timeTr) = time("Training") { linReg.fit(xTr, yTr) }
    val (_, timePr) = time("Prediction") { trainedLinReg.predict(xTe) }
    (timeTr, timePr)
  }

  val (timesTrTot, timesPrTot) = (1 to 100).foldLeft((List[Double](), List[Double]())) { case ((timesTr, timesPr), _) =>
    val (timeTr, timePr) = trainingPredictionTimes
    (timeTr :: timesTr, timePr :: timesPr)
  }

  val trainedLinReg = linReg.fit(xTr, yTr)
  val yTePred = trainedLinReg.predict(xTe)

  val timesTr = DenseVector(timesTrTot.toArray)
  val timesPr = DenseVector(timesPrTot.toArray)

  println(f"Training time: ${mean(timesTr)}%1.3fs (+/- ${2 * stddev(timesTr)}%1.3fs)")
  println(f"Prediction time: ${mean(timesPr)}%1.3fs (+/- ${2 * stddev(timesPr)}%1.3fs)")
  println(f"Test set rmse: ${rmse(yTe, yTePred)}%1.4f")
}
