import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/Datasets/Pokemon.csv")
#print(df.head())
#print(df.describe())

# 1. Comprension de datos
# 1.1. ¿Cuantos pokemon hay de cada tipo?
print(df["Type 1"].value_counts())
print(df["Type 2"].value_counts())

# 1.2. ¿Faltan valores en el conjunto de datos?
missing_values = df.isnull().sum()
print(missing_values)

# 1.3. ¿Cómo se distribuyen las estadisticas entre los diferentes pokemon?
plt.figure(figsize=(12, 6))
sns.boxplot(x="Type 1", y="HP", data=df)
plt.title("Distribución de HP por Tipo de Pokémon")
plt.xticks(rotation=90)
plt.xlabel("Tipo de Pokémon (Type 1)")
plt.ylabel("Puntos de Salud (HP)")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Type 1", y="Attack", data=df)
plt.title("Distribución de Ataque por Tipo de Pokémon")
plt.xticks(rotation=90)
plt.xlabel("Tipo de Pokémon (Type 1)")
plt.ylabel("Puntos de Ataque (Attack)")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Type 1", y="Defense", data=df)
plt.title("Distribución de Defensa por Tipo de Pokémon")
plt.xticks(rotation=90)
plt.xlabel("Tipo de Pokémon (Type 1)")
plt.ylabel("Puntos de defensa (Defense)")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Type 1", y="Sp. Atk", data=df)
plt.title("Distribución de Sp. Atk por Tipo de Pokémon")
plt.xticks(rotation=90)
plt.xlabel("Tipo de Pokémon (Type 1)")
plt.ylabel("Ataque especial (Sp. Atk)")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Type 1", y="Sp. Def", data=df)
plt.title("Distribución de Sp. Def por Tipo de Pokémon")
plt.xticks(rotation=90)
plt.xlabel("Tipo de Pokémon (Type 1)")
plt.ylabel("Defensa especial (Sp. Def)")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Type 1", y="Speed", data=df)
plt.title("Distribución de Speed por Tipo de Pokémon")
plt.xticks(rotation=90)
plt.xlabel("Tipo de Pokémon (Type 1)")
plt.ylabel("Velocidad (Speed)")
plt.show()

print(df.describe())

# 2. Preparacion de los datos
# 2.1. Limpieza: Manejar valores perdidos, si los hay.
if missing_values.any:
    df = df.drop('Type 2', axis=1)

missing_values = df.isnull().sum()
print(missing_values)

# 2.2. Transformacion: Codificar 'Type 1' en valores numericos.
df["Type 1"] = pd.factorize(df["Type 1"])[0]
print(df.describe())

# 3. Modelado
# 3.1. Para este proyecto, podríamos considerar usar un algoritmo de clasificación como Decision Trees o Random Forest.
num_features = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
cat_features = ["Type 1"]
preprocessor = make_column_transformer(
    (StandardScaler(), num_features),
    #(OneHotEncoder(), cat_features),
)
encoder = OneHotEncoder()
data_encoded = encoder.fit_transform(df[cat_features])

X = preprocessor.fit_transform(df)
y = df["Type 1"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf2 = RandomForestClassifier(n_estimators=105, random_state=42)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

# 4. Evaluacion
# 4.1. Usar una métrica como la precisión para evaluar qué tan bien funciona el modelo.
print("---------------Accuracy---------------")
accuracy = accuracy_score(y_test, y_pred)
print(f"Arbol de Decisión: {accuracy * 100:.2f}%")

accuracy = accuracy_score(y_test, y_pred2)
print(f"Random Forest: {accuracy * 100:.2f}%")
print("\n")

# 4.2. Considerar otros métodos, como matrices de confusión, para comprender más a fondo el rendimiento del modelo.
print("---------------Matrices de Confusión---------------")
print("Arbol de Decisión:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print("\n")

print("Random Forest:")
conf_matrix2 = confusion_matrix(y_test, y_pred2)
print(conf_matrix2)
print("\n")

# 5. Despliegue
# 5.1. Una vez que se tenga el modelo satisfactorio, se debe hacer la implementación donde los usuarios podrían ingresar las estadísticas de un Pokémon y predecir su `Type 1`.
def predicciones_usuario(modelo_arbol, modelo_forest, preprocessor):
    hp = float(input("Ingrese la vida de su pokemon: "))
    attack = float(input("Ingrese el ataque de su pokemon: "))
    defense = float(input("Ingrese la defensa de su pokemon: "))
    sp_atk = float(input("Ingrese el ataque especial de su pokemon: "))
    sp_def = float(input("Ingrese la defensa especial de su pokemon: "))
    speed = float(input("Ingrese la velocidad de su pokemon: "))
    
    pokemon = np.array([[hp, attack, defense, sp_atk, sp_def, speed]])


    pred_arbol = modelo_arbol.predict(pokemon)
    pred_forest = modelo_forest.predict(pokemon)

    # Mapeo de las predicciones a los nombres de los tipos
    type_mapping = {
        0: 'grass', 1: 'fire', 2: 'water', 3: 'bug', 4: 'normal',
        5: 'poison', 6: 'electric', 7: 'ground', 8: 'fairy', 9: 'fighting',
        10: 'psychic', 11: 'rock', 12: 'ghost', 13: 'ice', 14: 'dragon',
        15: 'dark', 16: 'steel', 17: 'flying'
    }

    tipo_predicho_arbol = type_mapping[pred_arbol[0]]
    tipo_predicho_forest = type_mapping[pred_forest[0]]

    print(f"El tipo de su pokemon segun la prediccion del arbol de decisión es: {pred_arbol[0]}")
    print(f"{tipo_predicho_arbol}")
    print(f"El tipo de su pokemon segun la predicción del random forest es: {pred_forest[0]}")
    print(f"{tipo_predicho_forest}")

while True:
    print("¿Desea realizar una predicción? (s/n)")
    respuesta = input()
    if respuesta == "s":
        predicciones_usuario(clf, clf2, preprocessor)
    elif respuesta == "n":
        break
    else:
        print("Respuesta no valida")

