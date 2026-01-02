import json
import time
import pandas as pd
from kafka import KafkaProducer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data/raw/marrakesh_temperature.csv")

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    event = {
        "date": row["time"],
        "tavg": row.get("tavg"),
        "tmin": row.get("tmin"),
        "tmax": row.get("tmax"),
        "prcp": row.get("prcp")
    }

    producer.send("weather-marrakesh", event)
    print("Sent:", event)
    time.sleep(0.2)

producer.flush()
producer.close()
