package com.picnicml.doddlemodel.utils

import breeze.linalg.DenseVector
import breeze.stats.{mean, stddev}
import com.picnicml.doddlemodel.metrics.Metric

object DisplayUtils {

  def displayResults(timesTr: DenseVector[Double],
                     timesPr: DenseVector[Double],
                     yTe: DenseVector[Double],
                     yTePred: DenseVector[Double],
                     metric: Metric): Unit = {
    println(f"Training time: ${mean(timesTr)}%1.3fs " + f"(+/- ${2 * stddev(timesTr)}%1.3fs)")
    println(f"Prediction time: ${mean(timesPr)}%1.3fs " + f"(+/- ${2 * stddev(timesPr)}%1.3fs)")
    println(f"Test set rmse: ${metric(yTe, yTePred)}%1.4f")
  }
}
