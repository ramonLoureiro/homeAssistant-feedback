import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from dotenv import load_dotenv, dotenv_values # type: ignore
from lib.load_data import LoadData # type: ignore

# Cargar variables desde .env
load_dotenv(override=True)
config = dotenv_values(".env")


# Priorizar valores de .env sobre variables de entorno
INFLUXDB_URL = config.get('INFLUXDB_URL') or os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN= config.get('INFLUXDB_TOKEN') or os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = config.get('INFLUXDB_ORG') or os.getenv('INFLUXDB_ORG')
INFLUXDB_BUCKET = config.get('INFLUXDB_BUCKET') or os.getenv('INFLUXDB_BUCKET')





def main():
    # Configuración de conexión a InfluxDB
    url=INFLUXDB_URL # Cambia según tu configuración
    token=INFLUXDB_TOKEN
    org=INFLUXDB_ORG
    bucket=INFLUXDB_BUCKET
    print(f"Conectando a InfluxDB en {url} {bucket}...")
    
    sensores = [
        'tuya_termometro_wifi',
        'shelly_blu_5480',
        'zigbee_sonoff_snzb02_01',
        'zigbee_heiman_hs3aq'
    ]

    # Crear instancia de LoadData
    loader = LoadData(url, token, org, bucket)
    # Obtener datos de temperatura
    df_temperatura = loader.obtener_datos('temperature', '°C', sensores)
    df_temperatura_preparados = loader.preparar_datos(df_temperatura, 'temperature',sensores) 


    # Obtener datos de humedad
    df_humedad = loader.obtener_datos("humidity", "%", sensores)
    df_humedad_preparados = loader.preparar_datos(df_humedad, 'humidity',sensores) 

    # Combinar datos
    df = loader.combinar_datos(df_temperatura_preparados, df_humedad_preparados)
    print(df.head())
    ultimo_valor = df.sort_values(by="timestamp", ascending=False).iloc[0]
    print(ultimo_valor)


if __name__ == '__main__':  
    main()