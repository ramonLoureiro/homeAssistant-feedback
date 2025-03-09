import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class PrediccionRandomForest:
    def __init__(self):
        self.name = "PrediccionRandomForest"
        self.model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_hum = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.entrenado = False
        self.df = None  # Atributo para almacenar el DataFrame de entrenamiento
        self.df_filtrado = None  # Atributo para almacenar el DataFrame filtrado


    def entrenar_modelo(self, df):
        """Entrena el modelo con datos históricos filtrados y con variables adicionales."""
        self.df = df.copy()  # Guardamos el DataFrame como atributo

        # Convertimos 'timestamp' a datetime y luego a valores numéricos (segundos desde Unix epoch)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['timestamp'] = self.df['timestamp'].astype('int64') // 10**9

        # Ordenamos por timestamp para calcular las diferencias
        self.df = self.df.sort_values(by='timestamp')

        # Creamos nuevas variables: diferencias de temperatura y humedad con 30 min antes
        self.df['temp_diff_30m'] = self.df['temperature'].diff().fillna(0)
        self.df['hum_diff_30m'] = self.df['humidity'].diff().fillna(0)



        # Filtramos por la hora actual
        now = pd.Timestamp.now()
        now_hour = now.hour
        df_filtrado = self.df[self.df['hora_dia'] == now_hour]



        # Si no hay suficientes datos, usamos todos los datos
        if df_filtrado.shape[0] < 5:
            df_filtrado = self.df

        # Selección de características
        X = df_filtrado[['timestamp', 'co2', 'hora_dia', 'dia_semana', 'semana', 'temp_diff_30m', 'hum_diff_30m']]
        y_temp = df_filtrado['temperature']
        y_hum = df_filtrado['humidity']

        # Normalizamos las características
        X_scaled = self.scaler.fit_transform(X)

        # Entrenamos los modelos de predicción para temperatura y humedad
        self.model_temp.fit(X_scaled, y_temp)
        self.model_hum.fit(X_scaled, y_hum)
        self.df_filtrado = df_filtrado  # Guardamos el DataFrame filtrado
        self.entrenado = True



    def predecir(self):
        """Realiza la predicción de temperatura y humedad utilizando las últimas tendencias."""
        if not self.entrenado:
            raise ValueError("El modelo no ha sido entrenado. Llama a 'entrenar_modelo' primero.")

        # Obtener la fecha y hora actual
        now = pd.Timestamp.now()
        now_timestamp = int(datetime.now().timestamp())
       
        now_hour = now.hour
        now_day = now.weekday()

        # Obtener la última observación real
        last_observation = self.df.iloc[-1]  # Tomamos la última fila del DataFrame real

        # Obtener la diferencia de temperatura y humedad respecto a la observación de hace 30 minutos
        temp_diff_30m = self.df['temperature'].iloc[-1] - self.df['temperature'].iloc[-2] if len(self.df) > 1 else 0
        hum_diff_30m = self.df['humidity'].iloc[-1] - self.df['humidity'].iloc[-2] if len(self.df) > 1 else 0

        # Crear un DataFrame con los datos actuales
        X_now = pd.DataFrame({
            'timestamp': [now_timestamp], 
            'co2': [500], 
            'hora_dia': [now_hour], 
            'dia_semana': [now_day], 
            'semana': [now.isocalendar()[1]],
            'temp_diff_30m': [temp_diff_30m],
            'hum_diff_30m': [hum_diff_30m]
        })

        # Normalizamos los datos actuales
        X_now_scaled = self.scaler.transform(X_now)

        # Realizamos las predicciones
        pred_temp = self.model_temp.predict(X_now_scaled)[0]
        pred_hum = self.model_hum.predict(X_now_scaled)[0]

        return {
            "timestamp":  datetime.now(),
            "temperature": round(pred_temp, 2),
            "humidity": round(pred_hum, 2)
        }
