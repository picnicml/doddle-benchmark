package com.picnicml.doddlemodel

trait TimingUtils {

  // code from http://biercoff.com/easily-measuring-code-execution-time-in-scala/
  def time[A](block: => A): A = {
    val start = System.currentTimeMillis
    val res = block
    val totalTime = System.currentTimeMillis - start
    println(s"Elapsed time: $totalTime ns")
    res
  }
}
