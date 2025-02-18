import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from influxdb_client import InfluxDBClient # type: ignore


class LoadData:
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
    
    def crear_query(self, param, units,columnas):
        # Generar condiciones de filtro
        condiciones = " or ".join([
            f"strings.containsStr(v: r.entity_id, substr: \"{columna +'_'+param}\")" 
            for columna in columnas
        ])
        
        query = f'''
        import "strings"

        from(bucket: "{self.bucket}")
        |> range(start: -30d)
        |> filter(fn: (r) => r["_measurement"] == "{units}")
        |> filter(fn: (r) => r["_field"] == "value")
        |> filter(fn: (r) => r["domain"] == "sensor")
        |> filter(fn: (r) => 
            {condiciones}
        )
        |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)            
        |> pivot(
            rowKey: ["_time"],
            columnKey: ["entity_id"], 
            valueColumn: "_value"
        )
        '''        
        return query

    def obtener_datos(self, param, unit, sensores):
        query = self.crear_query(param,unit,sensores)
        try:
            result = self.client.query_api().query_data_frame(query)
            
            if result is not None and not result.empty:
                # Eliminar columna _result si existe
                if '_result' in result.columns:
                    result = result.drop(columns=['_result'])
                
                # Renombrar columna de tiempo
                result = result.rename(columns={'_time': 'timestamp'})               
                result['timestamp'] = pd.to_datetime(result['timestamp'])
                return result
            else:
                print("No se encontraron datos")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error al obtener datos de temperatura: {e}")
            return pd.DataFrame()

    def preparar_datos(self, result,parametro,sensores):
        columnas = [sensor + '_' + parametro  for sensor in sensores]
        result[parametro] = result[columnas].mean(axis=1)
        # Añadir características de tiempo
        result['hora_dia'] = result['timestamp'].dt.hour
        result['dia_semana'] = result['timestamp'].dt.dayofweek
        return result

    def combinar_datos(self, df_temp, df_humedad):
        # Combinar DataFrames
        df_combinado = pd.merge(
            df_temp[['timestamp', 'temperature']], 
            df_humedad[['timestamp', 'humidity']], 
            on='timestamp', 
            how='inner'
        )
        
        # Añadir características de tiempo
        df_combinado['hora_dia'] = df_combinado['timestamp'].dt.hour
        df_combinado['dia_semana'] = df_combinado['timestamp'].dt.dayofweek
        return df_combinado