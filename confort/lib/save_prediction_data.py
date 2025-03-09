import os
from datetime import datetime
from influxdb_client import InfluxDBClient, Point # type: ignore
from dotenv import load_dotenv # type: ignore

#
# Clase para guardar los datos de predicción en InfluxDB
#
class SaveData:
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.org = org
        
        
    def saveValue(self,temperatura,humedad,location='casa'):
        """Escribe el dato en InfluxDB."""
        point = (
            Point("prediccion")  # Nombre de la "medida"
            .tag("sensor", "modelo_rf")  # Etiquetas para identificar la fuente de datos
            .field("temperature", temperatura)
            .field("humidity", humedad)
            .time(datetime.utcnow())            
        )

        with self.client.write_api() as write_api:
            write_api.write(bucket=self.bucket, org=self.org, record=point)
