import numpy as np
from sklearn.cluster import KMeans

data = np.array([[2,10], [2,5], [8,4], [5,8], [7,5], [6,4], [1,2], [4,9]],dtype = 'float32')
K = 3
centroids = np.array([[2,10], [5,8], [1,2]], dtype='float32')
kmeans = KMeans(n_clusters=K, init=np.array(centroids), n_init=1, max_iter =100, random_state=0)
kmeans.fit(data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
for i in range(K):
    cluster_points = data[labels == i].flatten().reshape(-1,2)
    print(f'\n Group {i+1}: {cluster_points}')
print('\n Centroids:')
print (centroids)