import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Cargar el conjunto de datos Iris
iris = load_iris()
# Valores
X = iris.data
# Etiquetas
y = iris.target

# Imprimir original
print("Iris original:")
print(X)

# Normalización
mean = np.mean(X, axis=0)
X -= mean
X /= np.std(X, axis=0)

# Imprimir el conjunto de datos después de la normalización
print("\nIris Normalizado:")
print(X)

# Aplicar PCA
n_components = 3  # Número de componentes principales
pca = PCA(n_components=n_components)
X_r = pca.fit_transform(X)

# Graficar los datos
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    ax.scatter(
        X_r[y == i, 0],
        X_r[y == i, 1],
        X_r[y == i, 2],
        color=color,
        alpha=0.8,
        lw=lw,
        label=target_name,
    )
ax.legend(loc="best", shadow=False, scatterpoints=1)
ax.set_title("PCA 3 componentes Iris")

plt.show()

# Aplicar PCA
n_components = 2  # Número de componentes principales
pca = PCA(n_components=n_components)
X_r = pca.fit_transform(X)


# Visualizar los datos reducidos a 2 dimensiones
plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA 2 Componentes Iris")

plt.show()
