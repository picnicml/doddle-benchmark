package com.picnicml.doddlemodel.scala

import breeze.linalg.DenseMatrix
import com.picnicml.doddlemodel.data.loadCsvDataset
import com.picnicml.doddlemodel.linear.SoftmaxClassifier
import org.openjdk.jmh.annotations.Benchmark

class JMH_Softmax {

  import SoftmaxFileData._

  @Benchmark
  def trainSoftmaxModel(): Unit = {
    model.fit(xTr, yTr)
  }
}

object SoftmaxFileData {
  val filePath = ""
  val data: DenseMatrix[Double] = loadCsvDataset(filePath, headerLine = false)
  val (xTr, yTr) = (data(0 until 80000, 0 to -2), data(0 until 80000, -1))
  val (xTe, yTe) = (data(80000 to -1, 0 to -2), data(80000 to -1, -1))
  val model = SoftmaxClassifier(1e-4)
}
