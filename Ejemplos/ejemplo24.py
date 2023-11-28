# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Presupuesto': [5000, 3500, 6000, 4500, 5500, 4200, 6800, 3200, 4900, 3800, 6100, 4300, 4700, 6900, 3300, 5000, 3600, 5200, 4400, 5600, 6100, 4100, 3400, 5000, 4800, 6700, 3500, 5300, 5100, 3700, 5900],
    'Clics': [12000, 8000, 15000, 10000, 11000, 9200, 16000, 7500, 10500, 8200, 14500, 9500, 11000, 15700, 7600, 10800, 8200, 12600, 8400, 14400, 9000, 7500, 9000, 7500, 11300, 10700, 15200, 8000, 11600, 12400, 8400],
    'Conversiones': [450, 320, 580, 390, 420, 350, 620, 290, 400, 315, 560, 365, 420, 605, 295, 410, 315, 480, 375, 430, 565, 345, 290, 435, 410, 590, 310, 445, 475, 320, 550]
}

df = pd.DataFrame(data)

# Seleccionar columnas del DataFrame
X = df[['Presupuesto', 'Clics']]
y = df['Conversiones']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones usando un modelo
y_pred = model.predict(X_test)

# Calcular el Mean Squared Error (MSE) y R-squared (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir los resultados
print(f"Predicción de Conversiones: {mse:.2f}")
print(f"Varianza: {r2:.2f}")

# Ejemplo de nuevos datos
X_new = np.array([[5500, 11500]])
y_new = model.predict(X_new)

print(f"Predicción de Conversiones: {y_new[0]:.2f}")
print(f"Varianza: {r2:.2f}")