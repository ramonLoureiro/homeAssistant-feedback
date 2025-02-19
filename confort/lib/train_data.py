
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

class EntrenaDatos:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None 

    def entrenar_modelo(self, df):
        # Características para el modelo
        self.feature_names = ['temperature', 'humidity', 'co2','hora_dia', 'dia_semana','semana']
        features = self.feature_names 
        print(features)        
        X = df[features]
        y = df['confort']
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


        # 3. Entrenar modelo
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        # 4. Evaluar modelo
        y_pred = knn.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # 5. Hacer predicciones
        nueva_muestra = [[24, 40]]  # Nueva entrada (temperatura y humedad)
        nueva_muestra_scaled = scaler.transform(nueva_muestra)
        prediccion = knn.predict(nueva_muestra_scaled)
        print("Predicción para la nueva muestra:", prediccion[0])