import os
import requests # type: ignore
import pandas as pd # type: ignore
import io
import numpy as np # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import PolynomialFeatures  # type: ignore
from influxdb_client import InfluxDBClient, Point # type: ignore
from influxdb_client.client.write_api import SYNCHRONOUS # type: ignore
from datetime import datetime
from dotenv import load_dotenv # type: ignore
import sys
import traceback



# Cargar variables desde .env
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")
INFLUXDB_BUCKET = 'NOAA_CO2_LEVELS'  # Nombre del bucket en InfluxDB

# URL del dataset de CO₂ en Mauna Loa (NOAA)
URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"

def download_data():
    try:
        response = requests.get(URL)
        response.raise_for_status()

        # Leer CSV correctamente evitando encabezados inesperados
        df = pd.read_csv(io.StringIO(response.text), skiprows=50, 
                         names=["year", "month", "decimal_date", "average", "deseasonalized", "ndays", "sdev", "unc"])

        # Eliminar filas vacías si existen
        df = df.dropna()

        # Convertir year y month a enteros para evitar errores
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)

        # Filtrar los datos anteriores a 1990
        df = df[df["year"] >= 1990]

        # Crear timestamps correctamente
        df["timestamp"] = df.apply(lambda row: datetime(int(row["year"]), int(row["month"]), 1), axis=1)

        # Convertir la columna "timestamp" a nanosegundos desde la época
        df["timestamp_ns"] = df["timestamp"].apply(lambda x: int(x.timestamp() * 1e9))  # Timestamp en nanosegundos

        return df

    except Exception as req_err:
        print(f"Error al descargar los datos: {req_err}")
        traceback.print_exc()
        return None


def extrapolate_data(df, years_ahead=20):
    """Extrapolación de los valores de CO₂ usando regresión polinómica"""
    df['timestamp_ns'] = df['timestamp_ns'] / 1e9  # Convertir a segundos

    X = df['timestamp_ns'].values.reshape(-1, 1)  # Convertir a array 2D
    y = df['deseasonalized'].values  # Valores de CO₂

    # Crear características polinómicas
    poly = PolynomialFeatures(degree=4)  # Ajusta el grado si es necesario
    X_poly = poly.fit_transform(X)

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_poly, y)

    # Generar nuevos timestamps
    last_timestamp = df['timestamp_ns'].max()
    future_timestamps = np.array([last_timestamp + (i * 365 * 24 * 60 * 60) for i in range(1, years_ahead + 1)])

    # Transformar los nuevos datos
    future_timestamps_poly = poly.transform(future_timestamps.reshape(-1, 1))
    predicted_co2 = model.predict(future_timestamps_poly)

    # Calcular decimal_date correctamente para los extrapolados
    base_timestamp = df['timestamp_ns'].min() / 1e9  # Usar el primer timestamp en segundos
    future_decimal_dates = (future_timestamps - base_timestamp) / (365.25 * 24 * 60 * 60) + df['decimal_date'].iloc[0]

    # Crear DataFrame con los valores extrapolados
    extrapolated_df = pd.DataFrame({
        "year": [datetime.utcfromtimestamp(ts).year for ts in future_timestamps],
        "month": [datetime.utcfromtimestamp(ts).month for ts in future_timestamps],
        "decimal_date": future_decimal_dates,
        "deseasonalized": predicted_co2,
        "timestamp_ns": future_timestamps * 1e9
    })

    print(f"Cantidad de puntos (filas) en extrapolated_df: {len(extrapolated_df)}")
    return extrapolated_df


def write_to_influxdb(df):
    # Establecer un límite de escritura para evitar sobrecarga
    BATCH_SIZE = 100
    
    try:
        # Crear cliente con timeout más largo
        client = InfluxDBClient(
            url=INFLUXDB_URL, 
            token=INFLUXDB_TOKEN, 
            org=INFLUXDB_ORG,
            timeout=30_000,  # 30 segundos
            connection_pool_maxsize=20
        )

        # Usar WriteAPI en modo SYNCHRONOUS para evitar problemas con futures
        write_api = client.write_api(write_options=SYNCHRONOUS)

        try:
            # Escribir en lotes para manejar grandes volúmenes de datos
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i:i+BATCH_SIZE]
                
                points = []
                for _, row in batch.iterrows():
                    point = Point("co2_levels") \
                        .tag("location", "Mauna_Loa") \
                        .field("average", float(row["average"])) \
                        .field("deseasonalized", float(row["deseasonalized"])) \
                        .time(int(row["timestamp_ns"]))  # Usar el timestamp en nanosegundos

                    points.append(point)

                # Escribir lote de puntos
                write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points)
                print(f"Procesado lote {i//BATCH_SIZE + 1}")

            print("Datos de CO₂ subidos exitosamente a InfluxDB")
            print(df.head())

        except Exception as write_err:
            print(f"Error al escribir en InfluxDB: {write_err}")
            traceback.print_exc()

        finally:
            # Cerrar explícitamente el cliente
            write_api.close()
            client.close()

    except Exception as client_err:
        print(f"Error al crear cliente InfluxDB: {client_err}")
        traceback.print_exc()

def main():
    try:
        # Descargar datos
        df = download_data()
        if df is not None and not df.empty:
            # Extrapolar datos para los próximos 10 años
            extrapolated_df = extrapolate_data(df, years_ahead=20)
            
            # Concatenar los datos originales con los extrapolados
            all_data_df = pd.concat([df, extrapolated_df], ignore_index=True)
            
            # Escribir los datos en InfluxDB
            write_to_influxdb(all_data_df)

            print(f"Datos de CO₂ (incluyendo extrapolados) subidos muy exitosamente a InfluxDB.")
            print(all_data_df.head())
            print(all_data_df.tail())
        else:
            print("No hay datos para procesar")

    except Exception as main_err:
        print(f"Error en la ejecución principal: {main_err}")
        traceback.print_exc()

def cleanup():
    """Limpiar recursos antes de salir"""
    import gc
    gc.collect()

# Punto de entrada principal
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error inesperado: {e}")
        traceback.print_exc()
    finally:
        cleanup()
        # Intentar forzar la salida
        try:
            sys.exit(0)
        except SystemExit:
            pass

