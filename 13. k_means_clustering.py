import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=3, random_state=42)

kmeans_model = KMeans(n_clusters=3, random_state=42)

kmeans_model.fit(X)

cluster_centers = kmeans_model.cluster_centers_
labels = kmeans_model.labels_

plt.scatter(X[:,0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', s= 50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.show()