package com.picnicml.doddlemodel.linear

import com.picnicml.doddlemodel.TimingUtils
import com.picnicml.doddlemodel.data.loadBostonDataset

object LinearRegressionTiming extends App with TimingUtils {
  val (x, y) = loadBostonDataset
  val model = LinearRegression(lambda = 1.5)

  val trainedModel = time {
    model.fit(x, y)
  }
}
