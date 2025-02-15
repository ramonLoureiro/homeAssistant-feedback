

import os
from datetime import datetime
from influxdb_client import InfluxDBClient, Point # type: ignore
from dotenv import load_dotenv # type: ignore

# Cargar variables desde .env
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")
INFLUXDB_DATATYPE = os.getenv("INFLUXDB_DATATYPE")


def get_last_temperatures(n=100):
    """Consulta los últimos n valores de temperatura en InfluxDB."""
    query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -7d)  // Última hora
        |> filter(fn: (r) => r._measurement == "{INFLUXDB_DATATYPE}" and r._field == "temperature")
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: {n})
    '''
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query_api = client.query_api()
        tables = query_api.query(query)
        temperatures = [record.get_value() for table in tables for record in table.records]
        return temperatures

def write_to_influx(temperature,location):
    """Escribe el dato en InfluxDB."""
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        with client.write_api() as write_api:
            point = Point(INFLUXDB_DATATYPE).tag("location", location).field("temperature", temperature).time(datetime.utcnow())
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            print(f"Escrito en InfluxDB: {temperature}°C")
