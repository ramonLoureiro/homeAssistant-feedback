from flask import Flask, jsonify

import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from dotenv import load_dotenv, dotenv_values # type: ignore
from confort.lib.load_data import LoadData # type: ignore
from confort.lib.model_random_forest import PrediccionRandomForest # type: ignore
from influxdb_client import InfluxDBClient # type: ignore


class PreparaData:
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.org = org
        self.url = url  
        self.token = token

        self.sensoresHTinterior = [
            'tuya_termometro_wifi',
            'shelly_blu_5480',
            'zigbee_sonoff_snzb02_01',
            'zigbee_heiman_hs3aq'
        ]


        self.sensoresCO2 = [
            'zigbee_heiman_hs3aq'
        ]

        self.sensoresHTexterior = [
            'aemet',
            'sbd_gran_via',
            'st_quirze_del_v_vallsuau_barcelona'
        ]




    def prepara_datos (self):
        print(f"Conectando a InfluxDB en {self.url} {self.bucket}...")

        # Crear instancia de LoadData
        loader = LoadData(self.url, self.token, self.org, self.bucket)
        # Obtener datos de temperatura
        df_temperatura = loader.obtener_datos('temperature', '°C', self.sensoresHTinterior)
        df_temperatura_preparados = loader.preparar_datos(df_temperatura, 'temperature',self.sensoresHTinterior) 

        
        df_temperatura_ext = loader.obtener_datos('temperature', '°C', self.sensoresHTexterior)
        df_temperatura_ext_preparados = loader.preparar_datos(df_temperatura_ext, 'temperature',self.sensoresHTexterior) 
        

        # Obtener datos de humedad
        df_humedad = loader.obtener_datos("humidity", "%", self.sensoresHTinterior)
        df_humedad_preparados = loader.preparar_datos(df_humedad, 'humidity',self.sensoresHTinterior) 

        df_co2 = loader.obtener_datos("co2", "ppm", self.sensoresCO2)
        df_co2_preparados = loader.preparar_datos(df_co2, 'co2',self.sensoresCO2) 


        df = loader.combinar_3_datos (df_temperatura_preparados,df_humedad_preparados,df_co2_preparados)

        df = df.sort_values(by='timestamp')

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)
        df[['temperature', 'humidity', 'co2']] = df[['temperature', 'humidity', 'co2']].round(2)

        ultimo_valor = df.sort_values(by="timestamp", ascending=False).iloc[0]

        df_final = df.dropna()
#        print(df_final)


        modelo = PrediccionRandomForest()
        modelo.entrenar_modelo(df_final)
        self.dataFrame = df_final
        # Hacer una predicción
        prediccion = modelo.predecir()
        prediccion['model'] = modelo.name
        prediccion['sensores'] = self.sensoresHTinterior + self.sensoresHTexterior + self.sensoresCO2
        prediccion['configuracion'] = self.url + ' ' + self.bucket
        print(prediccion)


        return (prediccion)



