import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
knn_labels = knn.predict(X)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(
    X[:, 0], X[:, 1], c=kmeans_labels, cmap="viridis", marker="o", edgecolor="k", s=50
)
axs[0].set_title("K-Means Clustering")
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")

axs[1].scatter(
    X[:, 0], X[:, 1], c=knn_labels, cmap="rainbow", marker="o", edgecolor="k", s=50
)
axs[1].set_title("KNN Classification")
axs[1].set_xlabel("Feature 1")
axs[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
