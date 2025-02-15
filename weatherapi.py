import os
import requests # type: ignore
from datetime import datetime
from lib.influxdb_utils import get_last_temperatures, write_to_influx
from dotenv import load_dotenv # type: ignore

# Cargar variables desde .env
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")


# Configuración
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
INFLUXDB_DATATYPE = os.getenv("INFLUXDB_DATATYPE")

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
        print("Error en API:", response.status_code, response.text)
        return None


# Ejecución
temperature = get_weather()
if temperature is not None:
    write_to_influx(temperature,{CITY})

# Consultar los últimos 10 valores
last_temperatures = get_last_temperatures(30)
print("Últimos 30 valores de temperatura:", last_temperatures)
