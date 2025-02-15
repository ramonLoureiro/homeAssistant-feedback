

import os
from influxdb_client import InfluxDBClient

# Cargar configuración desde variables de entorno
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://192.168.1.191:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "influxdb-j3Q!zK!RyjHWvMt-wvYq0UbRdRVqYAYPiTBDZkUNCYiyyU0D0dFXcdJ")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "home_assistant")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "ha_data")

def get_last_temperatures(n=10):
    """Consulta los últimos n valores de temperatura en InfluxDB."""
    query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -1h)  // Última hora
        |> filter(fn: (r) => r._measurement == "weather_data" and r._field == "temperature")
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: {n})
    '''
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query_api = client.query_api()
        tables = query_api.query(query)
        temperatures = [record.get_value() for table in tables for record in table.records]
        return temperatures
