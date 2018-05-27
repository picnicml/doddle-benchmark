package com.picnicml.doddlemodel

trait TimingUtils {

  // code from http://biercoff.com/easily-measuring-code-execution-time-in-scala/
  def time[A](what: String)(block: => A): (A, Double) = {
    val start = System.currentTimeMillis
    val res = block
    val totalTime = System.currentTimeMillis - start
    // println(s"$what time: ${totalTime / 1000.0}s")
    (res, totalTime / 1000.0)
  }
}
