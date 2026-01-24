from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import *

# Spark session
spark = SparkSession.builder \
    .appName("KafkaBatchConsumerLocal") \
    .getOrCreate()

# Read Kafka in BATCH mode
df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "weather-marrakesh") \
    .option("startingOffsets", "earliest") \
    .load()

# Convert Kafka value to string
df = df.selectExpr("CAST(value AS STRING) as json")

# Define schema of Meteostat JSON
schema = StructType([
    StructField("time", StringType()),
    StructField("tavg", DoubleType()),
    StructField("tmin", DoubleType()),
    StructField("tmax", DoubleType()),
    StructField("prcp", DoubleType()),
    StructField("wspd", DoubleType()),
])

# Parse JSON
df = df.select(from_json(col("json"), schema).alias("data")).select("data.*")

# Save locally as Parquet Data Lake
df.write.mode("overwrite").parquet("data_lake/weather_raw_parquet")

# Optional CSV export
df.toPandas().to_csv("data_lake/weather_raw.csv", index=False)

print("✅ Data saved locally in Parquet + CSV")
