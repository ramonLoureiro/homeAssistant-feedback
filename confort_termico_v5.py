import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ModeloConfort:
    def __init__(self):
        self.features = ['temp_int', 'hum_int', 'temp_ext', 'hum_ext']
        self.model_path = "modelos/modelo_confort.pkl"
        self.scaler_path = "modelos/scaler.pkl"
        self.model = None
        self.scaler = StandardScaler()
        self.cargar_modelo()

    def cargar_modelo(self):
        """Carga el modelo si existe, o inicializa uno nuevo"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("✅ Modelo de confort cargado.")
        else:
            print("⚠️ No se encontró un modelo previo. Se creará uno nuevo.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def entrenar_modelo(self, df):
        """Entrena o reentrena el modelo con nuevos datos"""
        if set(self.features + ['confort']).issubset(df.columns):
            X = df[self.features]
            y = df['confort']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model.fit(X_train_scaled, y_train)
            accuracy = self.model.score(X_test_scaled, y_test)

            print(f"✅ Modelo entrenado con precisión: {accuracy:.4f}")

            # Guardar modelo y scaler
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

        else:
            print("⚠️ El DataFrame no tiene las columnas necesarias.")

    def predecir_confort(self, datos_sensor):
        """Predice si el ambiente será confortable según tus hábitos"""
        datos_df = pd.DataFrame([datos_sensor], columns=self.features)
        datos_scaled = self.scaler.transform(datos_df)
        prediccion = self.model.predict(datos_scaled)[0]
        return "Confortable" if prediccion == 1 else "No confortable"
