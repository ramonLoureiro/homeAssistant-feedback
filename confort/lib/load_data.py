import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from influxdb_client import InfluxDBClient # type: ignore


class LoadData:
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.dias = 3
        self.media = '15m'
    
    def crear_query(self, param, units,columnas):
        # Generar condiciones de filtro
        condiciones = " or ".join([
            f"strings.containsStr(v: r.entity_id, substr: \"{columna +'_'+param}\")" 
            for columna in columnas
        ])
        
        condiciones = " or ".join([
            f"r.entity_id == \"{columna + '_' + param}\""
            for columna in columnas
        ])        
        query = f'''
        import "strings"

        from(bucket: "{self.bucket}")
        |> range(start: -{self.dias}d)
        |> filter(fn: (r) => r["_measurement"] == "{units}")
        |> filter(fn: (r) => r["_field"] == "value")
        |> filter(fn: (r) => r["domain"] == "sensor")
        |> filter(fn: (r) => 
            {condiciones}
        )
        |> aggregateWindow(every: {self.media}, fn: mean, createEmpty: false)            
        |> sort(columns: ["_time"], desc: false)
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
                result.drop(columns=['_start', '_stop','result','table','_measurement','_field','domain'], inplace=True)
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
        result['semana'] = result['timestamp'].dt.isocalendar().week
        return result


    def obtener_confort(self):
        query = f'''
        import "strings"
        from(bucket: "{self.bucket}")
        |> range(start: -{self.dias}d)
        |> filter(fn: (r) => r["_measurement"] == "confort")
        |> filter(fn: (r) => r["_field"] == "confort")
        |> filter(fn: (r) => r["location"] == "casa")
        |> sort(columns: ["_time"], desc: false)
        |> aggregateWindow(every: {self.media}, fn: mean, createEmpty: false)            
        |> pivot(
            rowKey: ["_time"],
            columnKey: ["_field"], 
            valueColumn: "_value"
        )
        '''        
        print (query)

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



    def combinar_3_datos(self, df_temp, df_humedad, df_co2):
        # Combinar DataFrames
        df_combinado1 = pd.merge(
            df_temp[['timestamp', 'temperature']], 
            df_humedad[['timestamp', 'humidity']], 
            on='timestamp', 
            how='inner'
        )
        df_combinado = pd.merge(
            df_combinado1, 
            df_co2[['timestamp', 'co2']], 
            on='timestamp', 
            how='inner'  # O 'left', 'right', 'outer' según tu necesidad
        )

        # Añadir características de tiempo
        df_combinado['hora_dia'] = df_combinado['timestamp'].dt.hour
        df_combinado['dia_semana'] = df_combinado['timestamp'].dt.dayofweek
        df_combinado['semana'] = df_combinado['timestamp'].dt.isocalendar().week
        return df_combinado



    def combinar_datos(self, df1, p1, df2, p2, how='inner'):
        """Combina dos DataFrames y añade características de tiempo."""

        # Combinar DataFrames
        df_combinado = pd.merge(
            df1[['timestamp', p1]],
            df2[['timestamp', p2]],
            on='timestamp',
            how=how  # Permite especificar el tipo de combinación
        )

        # Añadir características de tiempo
        df_combinado['hora_dia'] = df_combinado['timestamp'].dt.hour
        df_combinado['dia_semana'] = df_combinado['timestamp'].dt.dayofweek
        df_combinado['semana'] = df_combinado['timestamp'].dt.isocalendar().week.astype(int)  # Asegura tipo int
        return df_combinado
