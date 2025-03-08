from flask import Flask, jsonify

import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from dotenv import load_dotenv, dotenv_values # type: ignore
from confort.lib.load_data import LoadData # type: ignore
from confort.lib.model_random_forest import PrediccionRandomForest # type: ignore

# Cargar variables desde .env
load_dotenv(override=True)
config = dotenv_values("../confort/.env")

# Priorizar valores de .env sobre variables de entorno
INFLUXDB_URL = config.get('INFLUXDB_URL') or os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN= config.get('INFLUXDB_TOKEN') or os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = config.get('INFLUXDB_ORG') or os.getenv('INFLUXDB_ORG')
INFLUXDB_BUCKET = config.get('INFLUXDB_BUCKET') or os.getenv('INFLUXDB_BUCKET')


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

    ultimo_valor = df.sort_values(by="timestamp", ascending=False).iloc[0]

    df_final = df.dropna()
    print(df_final)


    modelo = PrediccionRandomForest()
    modelo.entrenar_modelo(df)

    # Hacer una predicción
    prediccion = modelo.predecir()
    print(prediccion)


    return (prediccion)




# Crear una instancia de la aplicación
app = Flask(__name__)

# Crear una ruta de prueba
@app.route('/')
def hello_world():
    return 'Hello, World! yuhu debug'

@app.route('/confort')
def confort():
    return 'Hello, World! confortable'



@app.route('/data')
def get_data():
    data = prepara_datos (sensoresHT_interior,sensoresHT_exterior,sensoresCO2)
#    return jsonify(ultimo_valor) 
    return data  # Devuelve JSON automáticamente




# Iniciar el servidor Flask

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
# Para ejecutar el servidor, ejecuta este script con python web.py
# y abre http://localhost:5000 en tu navegador.
# Verás el mensaje "Hello, World!".

