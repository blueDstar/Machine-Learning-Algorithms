import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Dữ liệu
n = np.array([[1,1], [3,2], [9,1], [3,7], [7,2], [9,7], [4,8], [8,3], [1,4]])

# Tính toán linkage để vẽ dendrogram
Z1 = sch.linkage(n, method='ward')
plt.figure(figsize=(6, 6))
sch.dendrogram(Z1)
plt.title("Hierarchical Clustering - Dendrogram (Ward Linkage)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Phân cụm bằng AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
labels = cluster.fit_predict(n)
print(labels)

# Vẽ scatter plot với kết quả phân cụm
fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(n[:, 0], n[:, 1], c=labels, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(), title="Nhóm", bbox_to_anchor=(1, 1))
ax.add_artist(legend)
plt.title("Hierarchical Clustering - Ward Linkage")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
