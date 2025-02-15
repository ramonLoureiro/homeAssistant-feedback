import os
from influxdb_client import InfluxDBClient, Point
from datetime import datetime

# 🔹 Configuración desde variables de entorno
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://192.168.1.191:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "influxdb-j3Q!zK!RyjHWvMt-wvYq0UbRdRVqYAYPiTBDZkUNCYiyyU0D0dFXcdJ")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "home_assistant")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "ha_data")

# Escribir datos en InfluxDB con manejo correcto de la conexión
with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
    with client.write_api() as write_api:
        point = Point("sensor_data") \
            .tag("location", "office") \
            .field("temperature", 23.5) \
            .field("humidity", 65) \
            .time(datetime.utcnow())

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        print(f"✅ Datos escritos en InfluxDB: {point}")


# Leer datos de InfluxDB    
with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
    query = f'from(bucket:"{INFLUXDB_BUCKET}") |> range(start: -1h)'
    query_api = client.query_api()
    tables = query_api.query(query, org=INFLUXDB_ORG)

    for table in tables:
        for record in table.records:
            print(record)