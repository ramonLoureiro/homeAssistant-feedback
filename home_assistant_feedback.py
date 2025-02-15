import os
import json
import time
import requests # type: ignore
import numpy as np # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from lib.influxdb_utils import get_last_temperatures
from dotenv import load_dotenv # type: ignore

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
    temperatures = get_last_temperatures(100)  # Últimos 10 valores
    if len(temperatures) < 2:
        return None  # No hay suficientes datos para predecir

    print(f"Temperaturas: {len(temperatures)}")

    X = np.arange(len(temperatures)).reshape(-1, 1)  # Tiempos [0, 1, 2, ...]
    y = np.array(temperatures)  # Temperaturas registradas
    
    model = LinearRegression().fit(X, y)
    future_time = np.array([[len(temperatures)]])  # Predicción para el siguiente punto
    predicted_temp = model.predict(future_time)[0]
    
    return round(predicted_temp, 2)

def update_sensor(value):
    url = f"{HOME_ASSISTANT_URL}/api/states/{ENTITY_ID}"
    data = json.dumps({"state": value})
    
    response = requests.post(url, headers=HEADERS, data=data)

    if response.status_code == 200:
        print(f"Sensor actualizado a {value}°C")
    else:
        print(f"Error al actualizar sensor: {response.text}")


start_time = time.time()  # Marca el inicio del script


predicted_temp = predict_temperature()
if predicted_temp:
    update_sensor(round(predicted_temp, 1))  # Actualizar sensor


end_time = time.time()  # Marca el final
elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido

print(f"Tiempo de ejecución: {elapsed_time:.6f} segundos")
