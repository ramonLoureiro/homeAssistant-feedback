import os
import requests
import pandas as pd
import io
from influxdb_client import InfluxDBClient, Point
from datetime import datetime
from dotenv import load_dotenv  # type: ignore

# Cargar variables desde .env
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

# URL del dataset de CO₂ en Mauna Loa (NOAA)
URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"

try:
    # Descargar datos
    response = requests.get(URL)
    response.raise_for_status()  # Esto lanzará una excepción si la solicitud no es exitosa

    # Leer CSV correctamente evitando encabezados inesperados
    df = pd.read_csv(io.StringIO(response.text), skiprows=50, names=["year", "month", "decimal_date", "average", "deseasonalized", "ndays", "sdev", "unc"])

    # Eliminar filas vacías si existen
    df = df.dropna()

    # Convertir year y month a enteros para evitar errores
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # Filtrar los datos anteriores a 1970
    df = df[df["year"] >= 1970]

    # Crear timestamps correctamente
    df["timestamp"] = df.apply(lambda row: datetime(int(row["year"]), int(row["month"]), 1), axis=1)

    # Convertir la columna "timestamp" a nanosegundos desde la época
    df["timestamp_ns"] = df["timestamp"].apply(lambda x: int(x.timestamp() * 1e9))  # Timestamp en nanosegundos

    # Usar el contexto 'with' para asegurar que la conexión se cierre correctamente
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        write_api = client.write_api()

        # Insertar cada fila en InfluxDB
        for _, row in df.iterrows():
            point = Point("co2_levels") \
                .tag("location", "Mauna_Loa") \
                .field("average", float(row["average"])) \
                .field("deseasonalized", float(row["deseasonalized"])) \
                .field("ndays", int(row["ndays"])) \
                .field("sdev", float(row["sdev"])) \
                .field("unc", float(row["unc"])) \
                .time(row["timestamp_ns"])

            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

        # Asegurarnos de que todos los puntos se hayan escrito antes de cerrar
        write_api.flush()

    print("Datos de CO₂ subidos exitosamente a InfluxDB")
    print(df.head())

except requests.exceptions.RequestException as req_err:
    print(f"Error al descargar los datos: {req_err}")
except pd.errors.ParserError as parse_err:
    print(f"Error al leer el CSV: {parse_err}")
except Exception as err:
    print(f"Ocurrió un error inesperado: {err}")
