import os
from datetime import datetime
from influxdb_client import InfluxDBClient, Point # type: ignore
from dotenv import load_dotenv # type: ignore


class SaveData:
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.org = org
        
    def saveValue(self,confort,location='casa'):
        """Escribe el dato en InfluxDB."""
        with self.client.write_api() as write_api:
            point = Point("confort").tag("location", location).field("confort", confort).time(datetime.utcnow())
            write_api.write(bucket=self.bucket, org=self.org, record=point)
            print(f"Escrito en InfluxDB: {confort}")