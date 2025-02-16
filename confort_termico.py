import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from influxdb_client import InfluxDBClient # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
import matplotlib.pyplot as plt # type: ignore
from dotenv import load_dotenv, dotenv_values # type: ignore

# Cargar variables desde .env
load_dotenv(override=True)
config = dotenv_values(".env")


# Priorizar valores de .env sobre variables de entorno
INFLUXDB_URL = config.get('INFLUXDB_URL') or os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN= config.get('INFLUXDB_TOKEN') or os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = config.get('INFLUXDB_ORG') or os.getenv('INFLUXDB_ORG')
INFLUXDB_BUCKET = config.get('INFLUXDB_BUCKET') or os.getenv('INFLUXDB_BUCKET')


class ComfortAnalytics:
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.model = None
        self.scaler = StandardScaler()
    

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

    def obtener_datos_temperatura(self, sensores):
        """
        Recuperar datos de temperatura de InfluxDB
        """
        query = self.crear_query('temperature','°C',sensores)
        try:
            result = self.client.query_api().query_data_frame(query)
            
            if result is not None and not result.empty:
                # Eliminar columna _result si existe
                if '_result' in result.columns:
                    result = result.drop(columns=['_result'])
                
                # Renombrar columna de tiempo
                result = result.rename(columns={'_time': 'timestamp'})
                
                # Convertir timestamp
                result['timestamp'] = pd.to_datetime(result['timestamp'])
                
                return result
            else:
                print("No se encontraron datos de temperatura")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error al obtener datos de temperatura: {e}")
            return pd.DataFrame()
    



    def obtener_datos_humedad(self, sensores):
        """
        Recuperar datos de humedad de InfluxDB
        """
        query = self.crear_query('humidity','%',sensores)
        
        try:
            result = self.client.query_api().query_data_frame(query)
            
            if result is not None and not result.empty:
                # Eliminar columna _result si existe
                if '_result' in result.columns:
                    result = result.drop(columns=['_result'])
                
                # Renombrar columna de tiempo
                result = result.rename(columns={'_time': 'timestamp'})
                
                # Convertir timestamp
                result['timestamp'] = pd.to_datetime(result['timestamp'])
                
                return result
            else:
                print("No se encontraron datos de humedad")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error al obtener datos de humedad: {e}")
            return pd.DataFrame()
    
    def preparar_datos_temperatura(self, result,sensores):
        columnas = [sensor + '_temperature' for sensor in sensores]
        # Calcular temperatura promedio
        result['temperatura'] = result[columnas].mean(axis=1)
        # Añadir características de tiempo
        result['hora_dia'] = result['timestamp'].dt.hour
        result['dia_semana'] = result['timestamp'].dt.dayofweek
        
        return result


    def preparar_datos_humedad(self, result,sensores):
        columnas = [sensor + '_humidity' for sensor in sensores]
        result['humedad'] = result[columnas].mean(axis=1)
        # Añadir características de tiempo
        result['hora_dia'] = result['timestamp'].dt.hour
        result['dia_semana'] = result['timestamp'].dt.dayofweek        
        return result


    def preparar_datos(self, df_temp, df_humedad):
        """
        Combinar datos de temperatura y humedad
        """
        # Combinar dataframes por timestamp
        df_combinado = pd.merge(
            df_temp[['timestamp', 'temperatura']], 
            df_humedad[['timestamp', 'humedad']], 
            on='timestamp', 
            how='inner'
        )
        
        # Añadir características de tiempo
        df_combinado['hora_dia'] = df_combinado['timestamp'].dt.hour
        df_combinado['dia_semana'] = df_combinado['timestamp'].dt.dayofweek
        
        # Definir confort (20-24°C, 40-60% humedad)
        df_combinado['confort'] = np.where(
            (df_combinado['temperatura'].between(17, 22)) & 
            (df_combinado['humedad'].between(40, 60)),
            1, 0
        )
        
        return df_combinado
    
    def entrenar_modelo(self, df):
        """
        Entrenar modelo de Random Forest
        """
        # Características para el modelo
        features = ['temperatura', 'humedad', 'hora_dia', 'dia_semana']
        X = df[features]
        y = df['confort']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Crear y entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test_scaled)
        print("Precisión del modelo:")
        print(accuracy_score(y_test, y_pred))
        print("\nInforme de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predecir_confort(self, nuevos_datos):
        """
        Predecir confort para nuevos datos
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Entrena primero.")
        
        # Escalar nuevos datos
        nuevos_datos_scaled = self.scaler.transform(nuevos_datos)
        
        # Predecir
        predicciones = self.model.predict(nuevos_datos_scaled)
        probabilidades = self.model.predict_proba(nuevos_datos_scaled)
        
        return predicciones, probabilidades
    

    
    def visualizar_importancia_caracteristicas(self):
        """
        Visualizar importancia de características
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        features = ['temperatura', 'humedad', 'hora_dia', 'dia_semana']
        importancia = self.model.feature_importances_
        
        plt.figure(figsize=(10,6))
        plt.bar(features, importancia)
        plt.title('Importancia de Características para Confort')
        plt.xlabel('Características')
        plt.ylabel('Importancia')
        plt.tight_layout()
        plt.show()

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



    # Crear instancia
    analizador = ComfortAnalytics(url, token, org, bucket)
    
    try:
        # Obtener datos
        df_temp = analizador.obtener_datos_temperatura(sensores)
        df_humedad = analizador.obtener_datos_humedad(sensores)
        # Preparar datos
        df_temp = analizador.preparar_datos_temperatura(df_temp,sensores)
        df_humedad = analizador.preparar_datos_humedad(df_humedad,sensores)
        df_combinado = analizador.preparar_datos(df_temp, df_humedad)
        
        # Entrenar modelo
        modelo = analizador.entrenar_modelo(df_combinado)
        
        # Visualizar importancia de características
        analizador.visualizar_importancia_caracteristicas()
        
        # Ejemplo de predicción (usa los primeros 5 registros)
        nuevos_datos = df_combinado[['temperatura', 'humedad', 'hora_dia', 'dia_semana']].iloc[:50]
        predicciones, probabilidades = analizador.predecir_confort(nuevos_datos)
        
        print("\nPredicciones de Confort:")
        for i, (pred, prob) in enumerate(zip(predicciones, probabilidades)):
            print(f"Muestra {i+1}: Confort={pred}, Probabilidad={prob.max():.2f}")
    
    except Exception as e:
        print(f"Error en el proceso: {e}")



if __name__ == "__main__":
    main()

