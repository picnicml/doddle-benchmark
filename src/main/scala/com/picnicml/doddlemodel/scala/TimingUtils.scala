package com.picnicml.doddlemodel.scala

trait TimingUtils {

  // code from http://biercoff.com/easily-measuring-code-execution-time-in-scala/
  def time[A](what: String)(block: => A): (A, Double) = {
    val start = System.currentTimeMillis
    val res = block
    val totalTime = System.currentTimeMillis - start
    (res, totalTime / 1000.0)
  }
}
