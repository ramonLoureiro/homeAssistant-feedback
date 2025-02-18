import datetime
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from influxdb_client import InfluxDBClient # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor # type: ignore
from sklearn.neural_network import MLPRegressor # type: ignore
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error # type: ignore

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
        self.feature_names = None 
        self.carpeta_modelos = "modelos"
        os.makedirs(self.carpeta_modelos, exist_ok=True)  # Crear carpeta si no existe
        print(f"Conectando a InfluxDB en {url} {bucket} modelos {self.carpeta_modelos}")    

    def modelos_existen(self):
        """Verifica si los modelos ya han sido guardados previamente en la carpeta 'modelos'."""
        return all(os.path.exists(os.path.join(self.carpeta_modelos, f"modelo_{m}.pkl")) for m in ["confort", "temperatura", "humedad"])

    def guardar_modelos(self):
        """Guarda los modelos en la carpeta 'modelos'."""
        joblib.dump(self.model_confort, "modelos/modelo_confort.pkl")
        joblib.dump(self.model_temp, "modelos/modelo_temperatura.pkl")
        joblib.dump(self.model_humedad, "modelos/modelo_humedad.pkl")
        joblib.dump(self.scaler, "modelos/scaler.pkl") # Guardamos el scaler
        joblib.dump(self.feature_names, "modelos/feature_names.pkl")
        print("Modelos guardados correctamente.")





    def cargar_modelos(self):
        """Carga los modelos desde la carpeta 'modelos'."""
        print("Cargando modelos existentes...")
        self.model_confort = joblib.load("modelos/modelo_confort.pkl")
        self.model_temp = joblib.load("modelos/modelo_temperatura.pkl")
        self.model_humedad = joblib.load("modelos/modelo_humedad.pkl")
        self.scaler = joblib.load("modelos/scaler.pkl")  # Cargamos el scaler
        self.feature_names = joblib.load("modelos/feature_names.pkl")
        self.model = self.model_confort
        print("Modelos cargados.")


    def definir_estado_confort(self, row):
        """
        Clasifica el nivel de confort en función de la temperatura y humedad.
        """
        if row['temperatura'] < 16:
            return 0  # Muy frío
        elif 17 <= row['temperatura'] <= 18 and 40 <= row['humedad'] <= 60:
            return 1  # Frío
        elif 19 <= row['temperatura'] <= 20 and 40 <= row['humedad'] <= 60:
            return 2  # Confort
        elif 23 < row['temperatura'] <= 21 and 40 <= row['humedad'] <= 60:
            return 3  # Cálido
        elif row['temperatura'] > 25:
            return 4  # Muy cálido
        else:
            return -1  # Indefinido (por si acaso hay valores fuera de estos rangos)



    def preparar_datos(self, df_temp, df_humedad):
        """
        Combina y etiqueta datos con categorías de confort.
        Elimina columnas duplicadas.
        """
        # Renombrar columnas para evitar duplicados
        df_temp = df_temp.rename(columns={'temperatura': 'temperatura_sensor'})
        df_humedad = df_humedad.rename(columns={'humedad': 'humedad_sensor'})

        # Combinar dataframes por timestamp
        df_combinado = pd.merge(
            df_temp[['timestamp', 'temperatura_sensor']], 
            df_humedad[['timestamp', 'humedad_sensor']], 
            on='timestamp', 
            how='inner'
        )
        
        # Renombrar para mantener compatibilidad
        df_combinado = df_combinado.rename(columns={
            'temperatura_sensor': 'temperatura',
            'humedad_sensor': 'humedad'
        })

        # Añadir características de tiempo
        df_combinado['hora_dia'] = df_combinado['timestamp'].dt.hour
        df_combinado['dia_semana'] = df_combinado['timestamp'].dt.dayofweek
        
        # Aplicar la clasificación de confort
        df_combinado['confort'] = df_combinado.apply(
            lambda row: self.definir_estado_confort(row), 
            axis=1
        )
        
        return df_combinado



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
        |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)            
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
    


    def preparar_datos_temperatura(self, result, sensores):
        # Renombrar columnas de temperatura
        columnas = [sensor + '_temperature' for sensor in sensores]
        
        # Calcular temperatura promedio
        result['temperatura'] = result[columnas].mean(axis=1)
        
        # Añadir características de tiempo
        result['hora_dia'] = result['timestamp'].dt.hour
        result['dia_semana'] = result['timestamp'].dt.dayofweek
        
        return result

    def preparar_datos_humedad(self, result, sensores):
        # Renombrar columnas de humedad
        columnas = [sensor + '_humidity' for sensor in sensores]
        
        # Calcular humedad promedio
        result['humedad'] = result[columnas].mean(axis=1)
        
        # Añadir características de tiempo
        result['hora_dia'] = result['timestamp'].dt.hour
        result['dia_semana'] = result['timestamp'].dt.dayofweek        
        
        return result


    def sandbox_preparar_datos(self, df_temp, df_humedad):
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
            (df_combinado['temperatura'].between(18, 22)) & 
            (df_combinado['humedad'].between(40, 65)),
            1, 0
        )
        
        return df_combinado
    
    def entrenar_modelo(self, df):
        """
        Entrenar modelo de Random Forest
        """
        # Características para el modelo
        self.feature_names = ['temperatura', 'humedad', 'hora_dia', 'dia_semana']
        features = self.feature_names 
        print(features)        
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
        print(classification_report(y_test, y_pred, zero_division=1))        
        return self.model



    def entrenar_modelos(self, df):
        """Entrenar modelos de confort y predicción"""

        features = self.feature_names or ['temperatura', 'humedad', 'hora_dia', 'dia_semana']
        print(features)        

        X = df[features]
        y_confort = df['confort']
        y_temp = df['temperatura'].shift(-1).dropna()
        y_humedad = df['humedad'].shift(-1).dropna()
        
        X_train, X_test, y_train_c, y_test_c = train_test_split(X[:-1], y_confort[:-1], test_size=0.2, random_state=42)
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X[:-1], y_temp, test_size=0.2, random_state=42)
        X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X[:-1], y_humedad, test_size=0.2, random_state=42)

        # 🔹 GUARDAR EL ORDEN DE LAS COLUMNAS ANTES DE ESCALAR
        self.feature_names = features
        
        # 🔹 ESCALAR LOS DATOS
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled_t = self.scaler.fit_transform(X_train_t)
        X_test_scaled_t = self.scaler.transform(X_test_t)
        X_train_scaled_h = self.scaler.fit_transform(X_train_h)
        X_test_scaled_h = self.scaler.transform(X_test_h)
        
        # 🔹 ENTRENAR MODELOS
        self.model_confort = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_confort.fit(X_train_scaled, y_train_c)
        y_pred_c = self.model_confort.predict(X_test_scaled)
        print("Precisión del modelo de confort:", accuracy_score(y_test_c, y_pred_c))
        
        # 🚨 ESTABLECER self.model IGUAL QUE self.model_confort
        self.model = self.model_confort

        self.model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_temp.fit(X_train_scaled_t, y_train_t)
        y_pred_t = self.model_temp.predict(X_test_scaled_t)
        print("Error medio absoluto (predicción temp):", mean_absolute_error(y_test_t, y_pred_t))
        
        self.model_humedad = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_humedad.fit(X_train_scaled_h, y_train_h)
        y_pred_h = self.model_humedad.predict(X_test_scaled_h)
        print("Error medio absoluto (predicción humedad):", mean_absolute_error(y_test_h, y_pred_h))

        self.evaluar_modelos(X_train_scaled, X_test_scaled, y_train_c, y_test_c) 
        self.guardar_modelos()




    def entrenar_modelos_incrementales(self, df_nuevo):
        """Reentrena los modelos con nuevos datos sin perder el conocimiento anterior."""
        print("🔄 Reentrenando modelos con nuevos datos...")

        # Usar los feature_names originales
        if self.feature_names is None:
            self.feature_names = ['temperatura', 'humedad', 'hora_dia', 'dia_semana']

        print("Feature names:", self.feature_names)

        # Asegurar que solo se usan las columnas originales
        X_nuevo = df_nuevo[self.feature_names]
        y_confort_nuevo = df_nuevo['confort']
        y_temp_nuevo = df_nuevo['temperatura'].shift(-1)
        y_humedad_nuevo = df_nuevo['humedad'].shift(-1)

        # Eliminar NaN para igualar dimensiones
        df_limpio = pd.concat([X_nuevo, y_temp_nuevo, y_humedad_nuevo], axis=1).dropna()

        X_nuevo = df_limpio[self.feature_names]
        y_temp_nuevo = df_limpio['temperatura']
        y_humedad_nuevo = df_limpio['humedad']
        y_confort_nuevo = y_confort_nuevo.loc[X_nuevo.index]  # Asegurar el mismo tamaño

        # Depuración
        print("Nuevos datos X shape:", X_nuevo.shape)
        print("Nuevos datos X columns:", X_nuevo.columns)
        print("Primeros registros:")
        print(X_nuevo.head())

        # Escalar datos 
        # Usar solo transform con las columnas en el mismo orden
        X_nuevo_scaled = self.scaler.transform(X_nuevo[self.feature_names])

        # Aumentar el número de árboles para aprendizaje incremental
        self.model_confort.n_estimators += 10
        self.model_temp.n_estimators += 10
        self.model_humedad.n_estimators += 10

        # Reentrenar usando warm_start=True
        self.model_confort.fit(X_nuevo_scaled, y_confort_nuevo)
        self.model_temp.fit(X_nuevo_scaled, y_temp_nuevo)
        self.model_humedad.fit(X_nuevo_scaled, y_humedad_nuevo)

        # Asegurar que self.model esté definido
        self.model = self.model_confort

        # Guardar modelos actualizados
        self.guardar_modelos()
        print("✅ Modelos reentrenados y guardados correctamente.")

    def diagnosticar_prediccion(self, nuevos_datos):
        """
        Método para diagnosticar problemas de predicción
        """
        print("🔍 Diagnóstico de Predicción")
        
        # Imprimir información de los modelos
        print("\n📊 Información de Modelos:")
        print(f"Modelo entrenado: {self.model}")
        print(f"Feature names guardados: {self.feature_names}")
        
        # Imprimir columnas de los nuevos datos
        print("\n📋 Columnas de Nuevos Datos:")
        print(nuevos_datos.columns)
        
        # Verificar tipos de datos
        print("\n📝 Tipos de Datos:")
        print(nuevos_datos.dtypes)
        
        # Verificar valores
        print("\n🔢 Primeros Registros:")
        print(nuevos_datos.head())
        
        # Verificar coincidencia de columnas
        columnas_coincidentes = [col for col in self.feature_names if col in nuevos_datos.columns]
        columnas_faltantes = [col for col in self.feature_names if col not in nuevos_datos.columns]
        
        print("\n✅ Columnas Coincidentes:")
        print(columnas_coincidentes)
        
        print("\n❌ Columnas Faltantes:")
        print(columnas_faltantes)
        
        # Intentar reconstruir DataFrame
        try:
            datos_filtrados = nuevos_datos[self.feature_names]
            print("\n🔧 Dataframe Reconstruido Exitosamente")
            print(datos_filtrados.head())
        except Exception as e:
            print(f"\n🚨 Error al reconstruir DataFrame: {e}")

        return columnas_coincidentes, columnas_faltantes




    # Función para entrenar y evaluar modelos
    def evaluar_modelos(self,X_train, X_test, y_train, y_test):
        modelos = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Red Neuronal": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000, random_state=42)
        }
        
        resultados = {}
        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            predicciones = modelo.predict(X_test)
            error = mean_absolute_error(y_test, predicciones)
            resultados[nombre] = error
            print(f"{nombre} - Error Medio Absoluto: {error:.4f}")
        
        return resultados






    def recomendar_accion(self, nuevos_datos):
        '''
        print("Recomendaciones:")
        print(nuevos_datos) ; print()

        dt = datetime.now()
        print('Datetime is:', dt)

        x = dt.weekday()
        print('Day of week:', x)
        h = dt.hour
        '''



        """Generar recomendaciones según predicciones"""
        nuevos_datos_scaled = self.scaler.transform(nuevos_datos)
        pred_temp = self.model_temp.predict(nuevos_datos_scaled)
        pred_humedad = self.model_humedad.predict(nuevos_datos_scaled)
        
        recomendaciones = []
        for temp, humedad in zip(pred_temp, pred_humedad):
            if temp < 18:
                recomendacion = "Encender calefacción"
            elif temp > 24:
                recomendacion = "Encender aire acondicionado"
            elif humedad > 60:
                recomendacion = "Encender deshumidificador"
            else:
                recomendacion= "Condiciones óptimas"
            print(f"Temperatura: {temp:.2f}°C, Humedad: {humedad:.2f}% {recomendacion}")
            recomendaciones.append(recomendacion) if recomendacion not in recomendaciones else None
        return recomendaciones


    def predecir_confort(self, nuevos_datos):
        """
        Predecir confort para nuevos datos con diagnóstico extendido
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Entrena primero.")

        # Diagnóstico detallado
        columnas_coincidentes, columnas_faltantes = self.diagnosticar_prediccion(nuevos_datos)

        # Intentar reconstruir DataFrame con columnas coincidentes
        if not columnas_coincidentes:
            raise ValueError("No hay columnas coincidentes para predecir")

        try:
            # Filtrar y ordenar columnas
            X_pred = nuevos_datos[self.feature_names]
            
            # Escalar datos
            nuevos_datos_scaled = self.scaler.transform(X_pred)
            
            # Predecir
            predicciones = self.model.predict(nuevos_datos_scaled)
            probabilidades = self.model.predict_proba(nuevos_datos_scaled)
            
            return predicciones, probabilidades

        except Exception as e:
            print(f"Error en predicción: {e}")
            raise




    def sandbox_predecir_confort(self, nuevos_datos):
        """
        Predecir confort para nuevos datos
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Entrena primero.")

        if self.feature_names:
            nuevos_datos = nuevos_datos[self.feature_names]  # 🔹 Reordenar columnas antes de transformar

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
    url=INFLUXDB_URL 
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
        if analizador.modelos_existen():
            analizador.cargar_modelos()
            analizador.entrenar_modelos_incrementales(df_combinado)
        else:
            analizador.entrenar_modelos(df_combinado)

        nuevos_datos = df_combinado[
            ['temperatura', 'humedad', 'hora_dia', 'dia_semana']
        ].iloc[:100]

        # Imprimir columnas antes de predecir
        print("\n📋 Columnas de Nuevos Datos:")
        print(nuevos_datos.columns)

        print("\n🔮 Prediciendo Confort:")
        predicciones, probabilidades = analizador.predecir_confort(nuevos_datos)
        
        print(predicciones)
        print(probabilidades)

    except Exception as e:
        print(f"Error en el proceso: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

