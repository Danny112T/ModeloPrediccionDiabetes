import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

np.random.seed(42)
n_samples = 1000
edades = np.random.randint(18, 70, n_samples)
salarios = np.random.randint(2000, 10000, n_samples)
tipo_produco = np.random.choice(["A", "B", "C"], n_samples)
interacciones = np.random.randint(1, 20, n_samples)

churn = np.where((salarios < 3000) | (interacciones > 15), 1, 0)
df = pd.DataFrame(
    {
        "edad": edades,
        "salario": salarios,
        "tipo_producto": tipo_produco,
        "interacciones": interacciones,
        "churn": churn,
    }
)
indices_faltantes = np.random.choice(
    df.index, size=int(0.05 * n_samples), replace=False
)  # no siempre se hace esto
df.loc[
    indices_faltantes, "salario"
] = np.nan  # ya que esto se limpia para simular la falta de datos (nan=not a number)
print(df.head())  # opcional

# Preprocesamiento
# llenar valores faltantes y eliminar registros con edades errorneas
df["salario"].fillna(df["salario"].mean(), inplace=True)
df = df[df["edad"] < 120]

# transformacion
# variables categoricas y numericas
cat_features = ["tipo_producto"]
num_features = ["edad", "salario", "interacciones"]

# crear transformadores para variables categoriacs y numericas
transformers = [
    ["num", StandardScaler(), num_features],
    ["cat", OneHotEncoder(drop="first"), cat_features],
]
preprocessor = ColumnTransformer(transformers)

# mineria de datos
# dividir los datos en entrenamiento y prueba
X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
