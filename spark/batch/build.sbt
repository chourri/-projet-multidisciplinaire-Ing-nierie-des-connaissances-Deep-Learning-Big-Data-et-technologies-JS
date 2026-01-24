name := "KafkaBatchConsumer"

version := "0.1"

scalaVersion := "2.12.18"   // keep this as is

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.4.1",
  "org.apache.spark" %% "spark-sql-kafka-0-10" % "3.4.1"
)
