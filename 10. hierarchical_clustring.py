import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4, random_state=42)

linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

clusters = AgglomerativeClustering(n_clusters=4, metric='euclidean',linkage='ward')
y_pred = clusters.fit_predict(X)

plt.scatter(X[0:,0], X[:,1], c=y_pred, cmap='viridis')
plt.title('Agglomerative Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()