import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/Datasets/housing.csv"
)

# Exploración de datos iniciales
#print(df.describe())
missing_values = df.isnull().sum()
if missing_values.any:
    df["total_bedrooms"].fillna(df["total_bedrooms"].mean(), inplace=True)

df["ocean_proximity"] = pd.factorize(df["ocean_proximity"])[0]

# Analisis de Correlación
df_2 = pd.DataFrame(df, columns=["median_income", "total_rooms", "median_house_value"])
correlation_matrix = df_2.corr()
"""plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True)
plt.show()"""


# Predicción del preció de viviendas
# normalización de los datos
df3 = df[["median_income", "total_rooms","housing_median_age","population","households","ocean_proximity","median_house_value"]]
scaler = StandardScaler()
df_standarized = scaler.fit_transform(df3)
df_standarized = pd.DataFrame(df_standarized, columns=df3.columns)
print(df_standarized)

# división del conjunto de datos
X , y = (df_standarized[["median_income", "total_rooms","housing_median_age","population","households","ocean_proximity"]], df_standarized["median_house_value"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo de regresión lineal
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)

# Evaluación del rendimiento del modelo
print("\n---------------Sin PCA---------------")
# Calcular el R^2 del modelo
r2 = r2_score(y_test, y_pred)
print("R^2:", r2)

# calcular MSE
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Coeficientes
print("Coefficients: ", linear_regression.coef_)

# Gráfico de dispersión del modelo de regresión lineal
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.show()

# Reducción de Dimensionalidad con Análisis de Componentes Principales (PCA)
fig1, axs1 = plt.subplots(3, 2, figsize=(15, 20))  # Crear una figura para los gráficos de PCA
fig1.suptitle('Gráficos de dispersión de las componentes principales')

fig2, axs2 = plt.subplots(3, 2, figsize=(15, 20))  # Crear una figura para los gráficos de regresión
fig2.suptitle('Gráficos de dispersión de los valores reales y predichos')

for i, n_components in enumerate(range(2, 7)):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df_standarized)

    # División del conjunto de datos (PCA)
    X_trainPCA, X_testPCA, y_trainPCA, y_testPCA = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    # Modelo de regresión lineal (PCA)
    linear_regressionPCA = linear_model.LinearRegression()
    linear_regressionPCA.fit(X_trainPCA, y_trainPCA)
    y_predPCA = linear_regressionPCA.predict(X_testPCA)

    # Evaluación del rendimiento del modelo (PCA)
    print(f"\n---------------PCA ({n_components} componentes)---------------")
    r2PCA = r2_score(y_testPCA, y_predPCA)
    print("R^2:", r2PCA)
    msePCA = mean_squared_error(y_testPCA, y_predPCA)
    print("MSE:", msePCA)

    # Gráfico de dispersión de las componentes principales
    ax1 = axs1[i//2, i%2]
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    ax1.set_xlabel('Primera Componente Principal')
    ax1.set_ylabel('Segunda Componente Principal')
    ax1.set_title(f'PCA con {n_components} componentes')

    # Gráfico de dispersión del modelo de regresión lineal (PCA)
    ax2 = axs2[i//2, i%2]
    ax2.scatter(y_testPCA, y_predPCA, color='blue')
    ax2.set_xlabel('Valores reales de PCA')
    ax2.set_ylabel('Valores predichos con el conjunto de datos PCA')
    ax2.plot([y_testPCA.min(), y_testPCA.max()], [y_testPCA.min(), y_testPCA.max()], color='red', linewidth=2)
    ax2.set_title(f'PCA con {n_components} componentes')

plt.tight_layout()
plt.show()

#Grafico de dispersion de componentes principales para 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_standarized)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.title('Gráfico de Dispersión de las Componentes Principales')
plt.colorbar(label='Valor Real (median_house_value)')
plt.show()