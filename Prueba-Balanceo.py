import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
#pip install imbalanced-learn

# Cargar el conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos para mejorar el rendimiento de K-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar sobremuestreo a los datos de entrenamiento para balancear las clases
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

# Modelo K-NN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_resampled, y_train_resampled)

# Modelo Regresión Logística
logreg_model = LogisticRegression()
logreg_model.fit(X_train_resampled, y_train_resampled)

# Modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_resampled, y_train_resampled)

# Realizar predicciones en el conjunto de prueba
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_logreg = logreg_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Calcular la precisión de los modelos
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Mostrar resultados
print("Precisión del modelo K-NN:", accuracy_knn)
print("Precisión del modelo Regresión Logística:", accuracy_logreg)
print("Precisión del modelo Random Forest:", accuracy_rf)

# Mostrar informe de clasificación
print("\nInforme de clasificación de K-NN:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

print("\nInforme de clasificación de Regresión Logística:")
print(classification_report(y_test, y_pred_logreg, target_names=iris.target_names))

print("\nInforme de clasificación de Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))
