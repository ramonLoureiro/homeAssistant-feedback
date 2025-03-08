import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

class ModeloConfort:
    def __init__(self, modelo_dir="modelos"):
        self.modelo_dir = modelo_dir
        os.makedirs(modelo_dir, exist_ok=True)

        self.features = ['temperatura', 'humedad', 'hora_dia', 'dia_semana']
        self.scaler = StandardScaler()
        self.model_confort = None
        self.model_temp = None
        self.model_humedad = None

        self.cargar_modelos()

    def entrenar_modelos(self, df):
        """Entrena o reentrena modelos desde cero con nuevos datos"""
        X = df[self.features]
        y_confort = df['confort']
        y_temp = df['temperatura'].shift(-1).dropna()
        y_humedad = df['humedad'].shift(-1).dropna()

        # División de datos
        X_train, X_test, y_train_c, y_test_c = train_test_split(X[:-1], y_confort[:-1], test_size=0.2, random_state=42)
        _, _, y_train_t, y_test_t = train_test_split(X[:-1], y_temp, test_size=0.2, random_state=42)
        _, _, y_train_h, y_test_h = train_test_split(X[:-1], y_humedad, test_size=0.2, random_state=42)

        # Normalización
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Modelos
        self.model_confort = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_humedad = RandomForestRegressor(n_estimators=100, random_state=42)

        self.model_confort.fit(X_train_scaled, y_train_c)
        self.model_temp.fit(X_train_scaled, y_train_t)
        self.model_humedad.fit(X_train_scaled, y_train_h)

        # Evaluación
        y_pred_c = self.model_confort.predict(X_test_scaled)
        y_pred_t = self.model_temp.predict(X_test_scaled)
        y_pred_h = self.model_humedad.predict(X_test_scaled)

        print("Precisión del modelo de confort:", accuracy_score(y_test_c, y_pred_c))
        print("Error medio absoluto (predicción temp):", mean_absolute_error(y_test_t, y_pred_t))
        print("Error medio absoluto (predicción humedad):", mean_absolute_error(y_test_h, y_pred_h))

        self.guardar_modelos()

    def guardar_modelos(self):
        """Guarda modelos y escalador"""
        joblib.dump(self.model_confort, os.path.join(self.modelo_dir, "modelo_confort.pkl"))
        joblib.dump(self.model_temp, os.path.join(self.modelo_dir, "modelo_temp.pkl"))
        joblib.dump(self.model_humedad, os.path.join(self.modelo_dir, "modelo_humedad.pkl"))
        joblib.dump(self.scaler, os.path.join(self.modelo_dir, "scaler.pkl"))
        print("✅ Modelos guardados.")

    def cargar_modelos(self):
        """Carga modelos y escalador si existen, sino los deja en None"""
        try:
            self.model_confort = joblib.load(os.path.join(self.modelo_dir, "modelo_confort.pkl"))
            self.model_temp = joblib.load(os.path.join(self.modelo_dir, "modelo_temp.pkl"))
            self.model_humedad = joblib.load(os.path.join(self.modelo_dir, "modelo_humedad.pkl"))
            self.scaler = joblib.load(os.path.join(self.modelo_dir, "scaler.pkl"))
            print("✅ Modelos cargados.")
        except FileNotFoundError:
            print("⚠️ No se encontraron modelos guardados, entrenar primero.")
