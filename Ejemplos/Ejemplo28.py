import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, centers= 3, random_state=42)

linkage = linkage(X, method='single')

dendrogram(linkage, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendograma de Agrupamiento Jerárquico')
plt.xlabel('Índice del Punto de Datos')
plt.ylabel('Distancia')
plt.show()