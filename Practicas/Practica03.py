# Se importan las librerias a utilizar
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = datasets.load_iris()  # se carga el dataset Iris en la variable Iris
X, y = (
    iris.data,
    iris.target,
)  # se asignan los datos de entrada y las etiquetas a las variables X e Y respectivamente
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # se dividen los datos en entrenamiento y prueba con un 80% y 20% respectivamente

## Se generan los modelos de regresión logistica y árbol de decisión
# Arbol de decision
clf = DecisionTreeClassifier(
    random_state=42
)  # se crea el clasificador de árbol de decisión
clf.fit(X_train, Y_train)  # se entrena el clasificador con los datos de entrenamiento
y_predTree = clf.predict(X_test)  # se realiza la predicción con los datos de prueba

# Regresión Logistica
regrLog = linear_model.LogisticRegression()  # se crea el modelo de regresión logistica
regrLog.fit(X_train, Y_train)  # se entrena el modelo con los datos de entrenamiento
y_predLog = regrLog.predict(X_test)  # se realiza la predicción con los datos de prueba


## Se imprimen los resultados de los modelos
print("----------Arbol de Decisión----------")
accuracy = accuracy_score(Y_test, y_predTree)  # se calcula la precisión del modelo
conf_matrix = confusion_matrix(Y_test, y_predTree)  # se calcula la matriz de confusión
print(f"Accuary: {accuracy * 100:.2f}%")  # se imprime la precisión del modelo
print(f"Confusion Matrix:\n{conf_matrix}")  # se imprime la matriz de confusión
print("\n")

print("----------Regresión Logistica----------")
print(
    "Coeficientes: \n", regrLog.coef_
)  # se imprimen los coeficientes de la regresión logistica
accuracy = accuracy_score(Y_test, y_predLog)  # se calcula la precisión del modelo
print(f"Accuary: {accuracy * 100:.2f}%")  # se imprime la precisión del modelo
print(
    "Confusion Matrix:\n", confusion_matrix(Y_test, y_predLog)
)  # se imprime la matriz de confusión
print("\n")

## Se grafican los resultados de los modelos
decision_tree_accuracy = clf.score( 
    X_test, Y_test
)  # se calcula la precisión del modelo de árbol de decisión
logistic_reg_accuracy = regrLog.score(
    X_test, Y_test
)  # se calcula la precisión del modelo de regresión logistica

## Se imprimen los resultados de los modelos
print(
    f"Precisión del modelo de Árbol de Decisiones: {decision_tree_accuracy:.2f}"
)  # se imprime la precisión del modelo de árbol de decisión
print(
    f"Precisión del modelo de Regresión Logistica: {logistic_reg_accuracy:.2f}"
)  # se imprime la precisión del modelo de regresión logistica


## Se define la función para realizar predicciones con los modelos
def predicciones_usuario(modelo_arbol_decision, modelo_regresion_log):
    # se solicitan los datos al usuario
    sepal_length = float(input("Ingrese la longitud del sépalo (cm): "))
    sepal_width = float(input("Ingrese el ancho del sépalo (cm): "))
    petal_length = float(input("Ingrese la longitud del pétalo (cm): "))
    petal_width = float(input("Ingrese el ancho del pétalo (cm): "))

    # se realiza la predicción con los datos ingresados por el usuario
    datos_usuario = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_regresion_log = modelo_regresion_log.predict(datos_usuario)
    pred_arbol_decision = modelo_arbol_decision.predict(datos_usuario)

    ## se imprimen los resultados de la predicción
    # se imprime el resultado de la predicción del modelo de regresión logistica
    print(f"Predicción del modelo de Regresión Logistica: {pred_regresion_log[0]}")
    if pred_regresion_log[0] == 0:
        print(
            "\tEn base a la predicción de regresion logistica la flor es de tipo Iris-setosa"
        )
    elif pred_regresion_log[0] == 1:
        print(
            "\tEn base a la predicción de regresion logistica la flor es de tipo Iris-versicolor"
        )
    elif pred_regresion_log[0] == 2:
        print(
            "\tEn base a la predicción de regresion logistica la flor es de tipo Iris-virginica"
        )
    else:
        print("Error en la predicción de regresion logistica")

    # se imprime el resultado de la predicción del modelo de árbol de decisión
    print(f"Predicción del modelo de Árbol de Decisión: {pred_arbol_decision[0]}")
    if pred_arbol_decision[0] == 0:
        print(
            "\tEn base a la predicción de árbol de decisión la flor es de tipo Iris-setosa"
        )
    elif pred_arbol_decision[0] == 1:
        print(
            "\tEn base a la predicción de árbol de decisión la flor es de tipo Iris-versicolor"
        )
    elif pred_arbol_decision[0] == 2:
        print(
            "\tEn base a la predicción de árbol de decisión la flor es de tipo Iris-virginica"
        )
    else:
        print("Error en la predicción del árbol de decisión")


## Se pregunta al usuario si desea realizar una nueva predicción
while True:
    print("¿Desea realizar una predicción? (s/n)")
    respuesta = input()
    if respuesta == "s":
        predicciones_usuario(clf, regrLog)
    elif respuesta == "n":
        break
    else:
        print("Opción no válida")
