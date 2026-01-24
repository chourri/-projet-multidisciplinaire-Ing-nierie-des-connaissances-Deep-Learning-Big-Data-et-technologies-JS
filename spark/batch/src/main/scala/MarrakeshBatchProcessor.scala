import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object MarrakeshBatchProcessor {

  def main(args: Array[String]): Unit = {

    // ===========================
    // 1️⃣ Spark Session
    // ===========================
    val spark = SparkSession.builder()
      .appName("KafkaBatchConsumerLocal")
      .master("local[*]")
      .config("spark.hadoop.fs.defaultFS", "file:///") // force local FS
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // ===========================
    // 2️⃣ Read Kafka in BATCH
    // ===========================
    val kafkaDF = spark.read
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "weather-marrakesh")
      .option("startingOffsets", "earliest")
      .load()

    // ===========================
    // 3️⃣ Convert Kafka Value to String
    // ===========================
    val dfString = kafkaDF.selectExpr("CAST(value AS STRING) as json")

    // ===========================
    // 4️⃣ Define Schema and Parse JSON
    // ===========================
    val schema = new StructType()
      .add("date", StringType)
      .add("tavg", DoubleType)
      .add("tmin", DoubleType)
      .add("tmax", DoubleType)
      .add("prcp", DoubleType)

    val dfParsed = dfString.select(from_json(col("json"), schema).alias("data"))
      .select("data.*")

    println("✅ Raw data parsed from Kafka")
    dfParsed.show(5, truncate = false)

    // ===========================
    // 5️⃣ Data Cleaning: convert to Double
    // ===========================
    val features = Seq("tavg", "tmin", "tmax", "prcp")

    val dfClean = features.foldLeft(dfParsed) { (df, f) =>
      df.withColumn(f, col(f).cast(DoubleType))
    }
      .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

    // ===========================
    // 6️⃣ Add dayofyear + seasonal features
    // ===========================
    val dfFeatures = dfClean
      .withColumn("dayofyear", dayofyear(col("date")))
      .withColumn("sin_doy", sin(col("dayofyear") * lit(2 * math.Pi) / lit(365)))
      .withColumn("cos_doy", cos(col("dayofyear") * lit(2 * math.Pi) / lit(365)))

    println("✅ Data cleaned and seasonal features added")
    dfFeatures.show(5, truncate = false)

    // ===========================
    // 7️⃣ Save cleaned data
    // ===========================
    dfFeatures.write.mode("overwrite").parquet("data_lake/weather_clean_parquet")
    dfFeatures.write.option("header", "true").mode("overwrite").csv("data_lake/weather_clean_csv")
    println("✅ Cleaned data saved in Parquet + CSV")

    // Stop Spark session
    spark.stop()
  }
}



//keep these columns:

//tavg tmin tmax prcp wspd pres


