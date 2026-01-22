import numpy as np
from sklearn.cluster import KMeans

n = np.array([15,15,16,19,19,20,20,21,22,28,35,40,41,42,43,44,60,61,65]).reshape(-1, 1)

# Số cụm
K = 2

kmeans = KMeans(n_clusters=K, init=np.array([16, 22]).reshape(-1, 1), n_init=1, max_iter =100, random_state=0)
kmeans.fit(n)

centroids = kmeans.cluster_centers_
clusters = kmeans.labels_

print(f"Tâm cụm cuối cùng: {n}")
# print(f"Phân cụm dữ liệu: {clusters}")
