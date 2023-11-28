import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, 
    accuracy_score, 
    recall_score, 
    confusion_matrix
)
from sklearn.datasets import make_classification

# Modelo de regresión logistica
class DannysLogisticRegressionModel:
    def __init__(self, learning_rate=0.01, max_iter=1000, convergence_criterion=1e-4):
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.convergence_criterion = convergence_criterion

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Inicializar los pesos con valores pequeños aleatorios
        self.weights = np.random.rand(X.shape[1])

        for iteration in range(self.max_iter):
            # Calcular la predicción lineal
            z = np.dot(X, self.weights)

            # Aplicar la función sigmoidea
            y_pred = self.sigmoid(z)

            # Calcular el error
            error = y - y_pred

            # Calcular el gradiente
            gradient = np.dot(X.T, error)

            # Actualizar los pesos
            self.weights += self.learning_rate * gradient

            # Criterio de convergencia
            if np.linalg.norm(gradient) < self.convergence_criterion:
                print(f"Convergencia alcanzada en la iteración {iteration + 1}.")
                break

    def predict(self, X_new):
        # Calcular la predicción lineal para nuevos datos
        z_new = np.dot(X_new, self.weights)

        # Aplicar la función sigmoidea
        y_pred_new = self.sigmoid(z_new)

        # Clasificar las salidas
        predicted_class = np.where(y_pred_new >= 0.5, 1, 0)

        return predicted_class


# Configurar la semilla para reproducibilidad
np.random.seed(117)

# Generar un dataset sintético para regresión logística
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=117)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generamos y entrenamos el modelo
RegLog = DannysLogisticRegressionModel()
RegLog.fit(X_train, y_train)
y_pred = RegLog.predict(X_test)

# Calcular la accuracy, precisión, recall, confusion matrix y 
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity}")

