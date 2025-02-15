import os
from influxdb_client import InfluxDBClient, Point # type: ignore
from datetime import datetime

# Configuraciones de conexión
url = os.getenv("INFLUXDB_URL")
token = os.getenv("INFLUXDB_TOKEN")
org = os.getenv("INFLUXDB_ORG")
bucket = os.getenv("INFLUXDB_BUCKET")

# Conectar a InfluxDB 2.x
client = InfluxDBClient(url=url, token=token, org=org)

# Escribir datos
write_api = client.write_api()

# Crear un punto de datos
point = Point("temperature") \
    .tag("location", "office") \
    .field("value", 23.5) \
    .time(datetime.utcnow())

# Escribir el punto en el bucket
write_api.write(bucket=bucket, org=org, record=point)

# Leer los datos
query = f'from(bucket:"{bucket}") |> range(start: -1h)'
query_api = client.query_api()

# Ejecutar la consulta
tables = query_api.query(query, org=org)

# Mostrar los resultados
for table in tables:
    for record in table.records:
        print(record)

# Cerrar cliente
client.close()

