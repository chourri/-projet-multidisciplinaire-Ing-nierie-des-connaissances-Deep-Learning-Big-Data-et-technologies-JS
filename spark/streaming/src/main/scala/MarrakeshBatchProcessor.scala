import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object MarrakeshStreamProcessor {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Marrakesh Weather Streaming")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // Read Kafka stream
    val kafkaDF = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "weather-marrakesh")
      .option("startingOffsets", "earliest")
      .load()

    // Define schema
    val schema = new StructType()
        .add("date", StringType)
        .add("tavg", DoubleType)
        .add("tmin", DoubleType)
        .add("tmax", DoubleType)
        .add("prcp", DoubleType)

    // Convert Kafka value to structured data
    val weatherDF = kafkaDF
      .selectExpr("CAST(value AS STRING)")
      .select(from_json(col("value"), schema).as("data"))
      .select("data.*")

    // Write streaming data to local folder as Parquet (simulated HDFS)
    val query = weatherDF.writeStream
      .outputMode("append")
      .format("parquet")
      .option("path", "/Users/chaimaehr/projects/marrakesh-weather-bigdata/hdfs-simulated/weather")
      .option("checkpointLocation", "/Users/chaimaehr/projects/marrakesh-weather-bigdata/hdfs-simulated/checkpoint")
      .start()

    query.awaitTermination()
  }
}
