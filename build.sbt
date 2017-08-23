name := "Imllib"

organization := "com.intel"

version := "0.0.1"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
    "org.apache.spark" % "spark-core_2.10" % "1.6.1",
    "org.apache.spark" % "spark-mllib_2.10" % "1.6.1",
    "org.specs2" %% "specs2-core" % "3.8.8" % "test"
)

resolvers += Resolver.sonatypeRepo("public")

scalacOptions in Test ++= Seq("-Yrangepos")

parallelExecution := false

publishTo := Some(Resolver.file("my_imllib",  new File(Path.userHome.absolutePath+"/.m2/repository")))
