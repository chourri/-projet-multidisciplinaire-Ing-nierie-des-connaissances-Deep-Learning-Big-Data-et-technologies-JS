from meteostat import Daily, Point
from datetime import datetime
import pandas as pd
import os

# Get project root (marrakesh-weather-bigdata)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Define Marrakesh
location = Point(31.63, -8.0)

# Fetch historical data
start = datetime(2010, 1, 1)
end = datetime(2024, 12, 31)

data = Daily(location, start, end).fetch()

# Save CSV
output_path = os.path.join(RAW_DATA_DIR, "marrakesh_temperature.csv")
data.to_csv(output_path)

print(f"Raw data saved at: {output_path}")
