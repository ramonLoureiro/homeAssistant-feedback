import os
import json
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from lib.influxdb_utils import get_last_temperatures
from dotenv import load_dotenv

# Cargar variables desde .env
load_dotenv()

HOME_ASSISTANT_URL = os.getenv("HOME_ASSISTANT_URL")
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")
ENTITY_ID = "sensor.temperatura_caldera"
HEADERS = {
    "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
    "Content-Type": "application/json"
}




def predict_temperature():
    """Predice la temperatura futura basándose en los últimos valores."""
    temperatures = get_last_temperatures(10)  # Últimos 10 valores
    if len(temperatures) < 2:
        return None  # No hay suficientes datos para predecir

    X = np.arange(len(temperatures)).reshape(-1, 1)  # Tiempos [0, 1, 2, ...]
    y = np.array(temperatures)  # Temperaturas registradas
    
    model = LinearRegression().fit(X, y)
    future_time = np.array([[len(temperatures)]])  # Predicción para el siguiente punto
    predicted_temp = model.predict(future_time)[0]
    
    return round(predicted_temp, 2)

def update_sensor(value):
    """Actualiza el sensor 'sensor.temperatura_manual' en Home Assistant."""
    url = f"{HOME_ASSISTANT_URL}/api/states/{ENTITY_ID}"
    data = json.dumps({"state": value})
    
    response = requests.post(url, headers=HEADERS, data=data)

    if response.status_code == 200:
        print(f"Sensor actualizado a {value}°C")
    else:
        print(f"Error al actualizar sensor: {response.text}")



predicted_temp = predict_temperature()
if predicted_temp:
    update_sensor(round(predicted_temp, 1))  # Actualizar sensor

