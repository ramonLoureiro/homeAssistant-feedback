import requests

# Sustituye esto con tu propia API Key
API_KEY = 'd0a78c77a34b4d55879212336251302'

# Puedes cambiar esto a la ciudad de tu preferencia
ciudad = 'Sabadell'

# URL de la API
url = f'http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={ciudad}'

# Realiza la solicitud GET a la API
response = requests.get(url)

# Verifica si la solicitud fue exitosa
if response.status_code == 200:
    data = response.json()
    
    # Acceder a algunos datos de la respuesta
    temperatura = data['current']['temp_c']  # Temperatura en grados Celsius
    condicion = data['current']['condition']['text']  # Condición del clima (soleado, lluvioso, etc.)
    humedad = data['current']['humidity']  # Humedad

    print(f'La temperatura en {ciudad} es de {temperatura}°C con condiciones {condicion}. Humedad: {humedad}%')
else:
    print(f"Error al obtener datos: {response.status_code}")

