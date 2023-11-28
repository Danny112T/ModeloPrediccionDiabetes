import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
)
from sklearn.datasets import make_classification


class DannysAdalineNRModel:
    def __init__(self, learning_rate=0.01, max_iter=1000, convergence_criterion=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.convergence_criterion = convergence_criterion
        self.weights = None

    def inicializar_pesos(self, n_features):
        return np.random.rand(n_features)
    
    def coste(self, y, y_pred):
        return (y - y_pred) ** 2
    
    def activacion(self, z):
        return z
    
    def fit(self, X, y):
        # Inicializar los pesos con valores pequeños aleatorios
        self.weights = self.inicializar_pesos(X.shape[1])

        # Para cada iteración hasta el máximo de iteraciones
        for _ in range(self.max_iter):
            # para cada muestra en el conjunto de datos de entrenamiento
            for i in range(X.shape[0]):
                # Calcula la entrada neta
                z = np.dot(X[i], self.weights)

                # Calcula la salida de la funcion de activacion
                y_pred = self.activacion(z)

                # Actualizar los pesos
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]

    def predict(self, X_new):
        # Calcular la predicción lineal para nuevos datos
        z_new = np.dot(X_new, self.weights)

        # Aplicar la función sigmoidea
        y_pred_new = self.activacion(z_new)

        # Clasificar las salidas
        predicted_class = np.where(y_pred_new >= 0.01, 1, 0)

        return predicted_class


# Configurar la semilla para reproducibilidad
np.random.seed(117)

# Generar un dataset sintético para regresión logística
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=117
)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Generamos y entrenamos el modelo
Adaline = DannysAdalineNRModel()
Adaline.fit(X_train, y_train)
y_pred = Adaline.predict(X_test)

# Calcular la accuracy, precisión, recall, confusion matrix y specificity
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity}")
