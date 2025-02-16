import pandas as pd # type: ignore
import numpy as np # type: ignore
from influxdb_client import InfluxDBClient # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
import matplotlib.pyplot as plt # type: ignore
from dotenv import load_dotenv # type: ignore

# Cargar variables desde .env
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

class ComfortAnalytics:
    def __init__(self, url, bucket, org, token):
        self.client = InfluxDBClient(
            url=url, 
            token=token, 
            org=org
        )
        self.bucket = bucket
        
    def fetch_comfort_data(self):
        """Recuperar datos de sensores"""
        query = f'''
        from(bucket:"{self.bucket}")
        |> range(start: -30d)
        |> filter(fn: (r) => 
            r._measurement == "temperature" or 
            r._measurement == "humidity" or
            r._measurement == "air_quality"
        )
        |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
        '''
        
        df = self.client.query_api().query_data_frame(query)
        return df
    
    def prepare_data(self, df):
        """Preprocesar datos"""
        features = [
            'temperature', 
            'humidity', 
            'air_quality', 
            'time_of_day', 
            'day_of_week'
        ]
        
        # Ingeniería de características
        df['time_of_day'] = pd.to_datetime(df['_time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['_time']).dt.dayofweek
        
        # Definir comfort como etiqueta objetivo
        df['comfort_level'] = np.where(
            (df['temperature'].between(20, 24)) & 
            (df['humidity'].between(40, 60)),
            1, 0
        )
        
        X = df[features]
        y = df['comfort_level']
        
        return X, y
    
    def train_comfort_model(self, X, y):
        """Entrenar modelo de red neuronal"""
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Modelo de red neuronal
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        model.fit(X_scaled, y, epochs=50, validation_split=0.2)
        
        return model, scaler
    
    def predict_comfort(self, model, scaler, new_data):
        """Predecir nivel de confort"""
        new_data_scaled = scaler.transform(new_data)
        predictions = model.predict(new_data_scaled)
        return predictions
    
    def generate_comfort_insights(self, predictions):
        """Generar recomendaciones de confort"""
        insights = []
        
        comfort_percentage = (predictions > 0.5).mean() * 100
        
        if comfort_percentage < 30:
            insights.append("Ambiente poco confortable")
            insights.append("Considera ajustar calefacción/climatización")
        
        elif comfort_percentage < 60:
            insights.append("Confort intermedio")
            insights.append("Pequeños ajustes pueden mejorar la sensación térmica")
        
        else:
            insights.append("Excelente confort térmico")
            insights.append("Condiciones ambientales óptimas")
        
        return insights
    
    def visualize_comfort(self, df):
        """Visualizar datos de confort"""
        plt.figure(figsize=(12,6))
        
        plt.subplot(2,2,1)
        plt.title('Temperatura')
        plt.plot(df['_time'], df['temperature'])
        
        plt.subplot(2,2,2)
        plt.title('Humedad')
        plt.plot(df['_time'], df['humidity'])
        
        plt.tight_layout()
        plt.show()

# Uso del sistema
def main():
    comfort_analyzer = ComfortAnalytics(
        url   = INFLUXDB_URL,
        bucket= INFLUXDB_BUCKET, 
        org   = INFLUXDB_ORG, 
        token = INFLUXDB_TOKEN
    )
    
    # Recuperar datos
    df = comfort_analyzer.fetch_comfort_data()
    
    # Preparar datos
    X, y = comfort_analyzer.prepare_data(df)
    
    # Entrenar modelo
    model, scaler = comfort_analyzer.train_comfort_model(X, y)
    
    # Predecir confort
    new_data = X.iloc[:10]  # Ejemplo con primeros 10 registros
    predictions = comfort_analyzer.predict_comfort(model, scaler, new_data)
    
    # Generar insights
    insights = comfort_analyzer.generate_comfort_insights(predictions)
    
    # Visualizar
    comfort_analyzer.visualize_comfort(df)
    
    # Mostrar insights
    for insight in insights:
        print(insight)

if __name__ == "__main__":
    main()

