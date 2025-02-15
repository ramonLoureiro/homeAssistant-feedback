

import os
from influxdb_client import InfluxDBClient

from dotenv import load_dotenv

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
