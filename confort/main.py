import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from dotenv import load_dotenv, dotenv_values # type: ignore
from lib.load_data import LoadData # type: ignore
from lib.train_data import EntrenaDatos # type: ignore

# Cargar variables desde .env
load_dotenv(override=True)
config = dotenv_values(".env")


# Priorizar valores de .env sobre variables de entorno
INFLUXDB_URL = config.get('INFLUXDB_URL') or os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN= config.get('INFLUXDB_TOKEN') or os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = config.get('INFLUXDB_ORG') or os.getenv('INFLUXDB_ORG')
INFLUXDB_BUCKET = config.get('INFLUXDB_BUCKET') or os.getenv('INFLUXDB_BUCKET')


def prepara_datos (sensoresHTinterior,sensoresHTexterior,sensoresCO2):
    # Configuración de conexión a InfluxDB
    url=INFLUXDB_URL # Cambia según tu configuración
    token=INFLUXDB_TOKEN
    org=INFLUXDB_ORG
    bucket=INFLUXDB_BUCKET
    print(f"Conectando a InfluxDB en {url} {bucket}...")

    # Crear instancia de LoadData
    loader = LoadData(url, token, org, bucket)
    # Obtener datos de temperatura
    df_temperatura = loader.obtener_datos('temperature', '°C', sensoresHTinterior)
    df_temperatura_preparados = loader.preparar_datos(df_temperatura, 'temperature',sensoresHTinterior) 

    
    df_temperatura_ext = loader.obtener_datos('temperature', '°C', sensoresHTexterior)
    df_temperatura_ext_preparados = loader.preparar_datos(df_temperatura_ext, 'temperature',sensoresHTexterior) 
    

    df_combinado = pd.merge(
        df_temperatura, 
        df_temperatura_ext, 
        on='timestamp', 
        how='left', 
        suffixes=('_int', '_ext')  # Añade sufijos para diferenciar
    )
    print(df_combinado)

    # Obtener datos de humedad
    df_humedad = loader.obtener_datos("humidity", "%", sensoresHTinterior)
    df_humedad_preparados = loader.preparar_datos(df_humedad, 'humidity',sensoresHTinterior) 

    df_co2 = loader.obtener_datos("co2", "ppm", sensoresCO2)
    df_co2_preparados = loader.preparar_datos(df_co2, 'co2',sensoresCO2) 


    df = loader.combinar_3_datos (df_temperatura_preparados,df_humedad_preparados,df_co2_preparados)

    df = df.sort_values(by='timestamp')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    df[['temperature', 'humidity', 'co2']] = df[['temperature', 'humidity', 'co2']].round(2)

    print(df)
    ultimo_valor = df.sort_values(by="timestamp", ascending=False).iloc[0]
    print(ultimo_valor)



    df_confort = loader.obtener_confort()
    df_confort['timestamp'] = pd.to_datetime(df_confort['timestamp'])

    df_confort = df_confort.sort_values(by='timestamp')    
 #   df_confort['timestamp'] = pd.to_datetime(df_confort['timestamp'])
    df_confort.drop(columns=['_start', '_stop','result','table'], inplace=True)
    df_confort['timestamp'] = df_confort['timestamp'].dt.tz_convert(None)
    df_confort[['confort']] = df_confort[['confort']].astype(int)


#    df_confort = df_confort.dropna()

    print('-------------') 
    print(df_confort)
    print('-------------') 

    # 2. Localiza los timestamps a UTC (esto es importante si ya tienen información de zona horaria)
    df_total = pd.merge(df, df_confort, on='timestamp', how='left')
#    df_total.drop(columns=['_start', '_stop','result','table'], inplace=True)
    df_final = df_total.dropna()


    print(df_final)



def main():
    
    sensoresHT_interior = [
        'tuya_termometro_wifi',
        'shelly_blu_5480',
        'zigbee_sonoff_snzb02_01',
        'zigbee_heiman_hs3aq'
    ]


    sensoresCO2 = [
        'zigbee_heiman_hs3aq'
    ]

    sensoresHT_exterior = [
        'aemet',
        'sbd_gran_via',
        'st_quirze_del_v_vallsuau_barcelona'
    ]


    prepara_datos (sensoresHT_interior,sensoresHT_exterior,sensoresCO2)


#    entrenador = EntrenaDatos()
#    entrenador.entrenar_modelo(df)


if __name__ == '__main__':  
    main()