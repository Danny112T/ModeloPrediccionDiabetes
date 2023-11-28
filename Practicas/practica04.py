import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Sloth Data.csv")

# 1. Preprocesamiento
# 1.1.  Limpieza de Datos:
# 1.1.1. Identificar y manejar valores faltantes.
missing_values = df.isnull().sum()
if missing_values.any:
    df["claw_length_cm"].fillna(df["claw_length_cm"].mean(), inplace=True)
    df["size_cm"].fillna(df["size_cm"].mean(), inplace=True)
    df["weight_kg"].fillna(df["weight_kg"].mean(), inplace=True)
    df["tail_length_cm"].fillna(df["tail_length_cm"].mean(), inplace=True)

# 1.1.2. Detectar y tratar posibles valores atípicos o anomalías en las dimensiones.
df = df[(df["claw_length_cm"] < 13) & (df["claw_length_cm"] > 0)]
df = df[(df["size_cm"] < 80) & (df["size_cm"] > 0)]
df = df[(df["weight_kg"] < 10) & (df["weight_kg"] > 0)]
df = df[(df["tail_length_cm"] < 10) & (df["tail_length_cm"] > 0)]

print(df.describe())
# 1.2 transformacion de datos
# 1.2.1. Codificar variables categóricas (como la clasificación de peligro) si es necesario.
df["endangered"] = pd.factorize(df["endangered"])[0]

# 2. Transformación:
# 2.1. Generar nuevas características si es posible, como un índice de tamaño basado en varias dimensiones.
# Calcular la media de las cuatro columnas para cada fila
df["mean_value"] = df[
    ["claw_length_cm", "size_cm", "weight_kg", "tail_length_cm"]
].mean(axis=1)
# Generar la nueva columna categórica con cut()
df["category"] = pd.cut(df["mean_value"], bins=4, labels=["0", "1", "2", "3"])

# 2.2. Considerar la normalización de dimensiones si se usan modelos basados en distancia (como K-Means o Min-Max Scaling).
num_features = ["claw_length_cm", "size_cm", "weight_kg", "tail_length_cm"]
transformers = [
    ["num", StandardScaler(), num_features],
]
preprocessor = ColumnTransformer(transformers)

# 3. Mineria de Datos:
# 3.1. Analisis Exploratiorio
# 3.1.1. Visualizar la distribución de las dimensiones.
df[num_features].hist(bins=30, figsize=(10,7))
plt.show()
# 3.1.2. Analizar las diferencias de tamaño entre perezosos de dos y tres dedos.
sns.boxplot(x='specie', y='mean_value', data=df)
plt.show()
# 3.1.3 Examinar la relación entre las dimensiones y la clasificación de peligro.
plt.scatter(df['endangered'], df['mean_value'])
plt.xlabel('En peligro')
plt.ylabel('Tamaño promedio')
plt.show()

# 3.2. Modelado
# 3.2.1. Clasifica entre perezosos de dos o tres dedos, entrenar un modelo de clasificación (regresión logística o árbol de decisiones).
X, y = (
    df[["claw_length_cm", "size_cm", "weight_kg", "tail_length_cm"]],
    df['specie']
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Arbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_predTree = clf.predict(X_test)

# Regresión Logistica
regrLog = linear_model.LogisticRegression()  # se crea el modelo de regresión logistica
regrLog.fit(X_train, y_train)  # se entrena el modelo con los datos de entrenamiento
y_predLog = regrLog.predict(X_test)  # se realiza la predicción con los datos de prueba

# 3.2.2. Validar el modelo utilizando una división de entrenamiento-prueba o validación cruzada.
print("----------Arbol de Decisión----------")
accuracy = accuracy_score(y_test, y_predTree)
conf_matrix = confusion_matrix(y_test, y_predTree)
print(f"Accuary: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")
print("\n")

print("----------Regresión Logistica----------")
print(
    "Coeficientes: \n", regrLog.coef_
)
accuracy = accuracy_score(y_test, y_predLog)
print(f"Accuary: {accuracy * 100:.2f}%")
print(
    "Confusion Matrix:\n", confusion_matrix(y_test, y_predLog)
)
print("\n")

decision_tree_accuracy = clf.score(X_test, y_test)
logistic_reg_accuracy = regrLog.score(X_test, y_test)
print(f"Precisión del modelo de Árbol de Decisiones: {decision_tree_accuracy:.2f}")
print(f"Precisión del modelo de Regresión Logistica: {logistic_reg_accuracy:.2f}")

from sklearn.model_selection import cross_val_score

# Modelo de árbol de decisión
decision_tree_scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation
print("Precisión del modelo de Árbol de Decisiones (validación cruzada):", decision_tree_scores)
print("Precisión promedio del modelo de Árbol de Decisiones:", decision_tree_scores.mean())

# Modelo de regresión logística
logistic_reg_scores = cross_val_score(regrLog, X, y, cv=5)  # 5-fold cross-validation
print("Precisión del modelo de Regresión Logística (validación cruzada):", logistic_reg_scores)
print("Precisión promedio del modelo de Regresión Logística:", logistic_reg_scores.mean())
