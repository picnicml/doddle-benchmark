

lazy val root = (project in file("."))
  .settings(
    name := "doddle-benchmark",
    organization := "com.picnicml",
    version := "0.0.0",
    scalaVersion := "2.12.4",
    libraryDependencies ++= "com.picnicml" %% "doddle-model" % "0.0.0" ::
      "org.slf4j" % "slf4j-nop" % "1.7.25" :: Nil
  )
