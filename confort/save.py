import os
import pandas as pd # type: ignore
from dotenv import load_dotenv, dotenv_values # type: ignore
from lib.save_data import SaveData # type: ignore
import argparse

# Cargar variables desde .env
load_dotenv(override=True)
config = dotenv_values(".env")


# Priorizar valores de .env sobre variables de entorno
INFLUXDB_URL = config.get('INFLUXDB_URL') or os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN= config.get('INFLUXDB_TOKEN') or os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = config.get('INFLUXDB_ORG') or os.getenv('INFLUXDB_ORG')
INFLUXDB_BUCKET = config.get('INFLUXDB_BUCKET') or os.getenv('INFLUXDB_BUCKET')





def main():
    # Configuración de conexión a InfluxDB
    url=INFLUXDB_URL # Cambia según tu configuración
    token=INFLUXDB_TOKEN
    org=INFLUXDB_ORG
    bucket=INFLUXDB_BUCKET
    print(f"Conectando a InfluxDB en {url} {bucket}...")
    

    # Crear instancia de LoadData

    parser = argparse.ArgumentParser(description="Leer valores de confort.")

    # Definir los argumentos esperados
    parser.add_argument('--confort', type=int, required=True, help="Valor de confort 0 a 10")

    # Parsear los argumentos
    args = parser.parse_args()

    # Acceder a los valores
    confort = args.confort

    # Imprimir los valores
    print(f"Confort: {confort}")    
    saver = SaveData(url, token, org, bucket)
    saver.saveValue (confort)


if __name__ == '__main__':  
    main()


