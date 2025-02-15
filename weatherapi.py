import os
import requests
from influxdb_client import InfluxDBClient, Point
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from lib.influxdb_utils import get_last_temperatures
from dotenv import load_dotenv

# Cargar variables desde .env
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")


# 🔗 Configuración
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CITY = os.getenv("CITY")


def get_weather():
    """Obtiene la temperatura actual de WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={CITY}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data["current"]["temp_c"]
    else:
        print(url);
        print("❌ Error en API:", response.status_code, response.text)
        return None

def write_to_influx(temperature):
    """Escribe el dato en InfluxDB."""
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        with client.write_api() as write_api:
            point = Point("weather_data").tag("location", CITY).field("temperature", temperature).time(datetime.utcnow())
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            print(f"✅ Escrito en InfluxDB: {temperature}°C")







# 🔁 Ejecución
temperature = get_weather()
if temperature is not None:
    write_to_influx(temperature)

# 📊 Consultar los últimos 10 valores
last_temperatures = get_last_temperatures(10)
print("🌡️ Últimos 10 valores de temperatura:", last_temperatures)
