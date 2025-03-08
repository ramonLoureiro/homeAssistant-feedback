import os
from dotenv import load_dotenv

# Imprimir ruta actual
print("Directorio de trabajo actual:", os.getcwd())

# Imprimir todas las variables de entorno
print("\nTodas las variables de entorno:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

# Cargar variables de entorno de manera verbose
load_dotenv(verbose=True)

