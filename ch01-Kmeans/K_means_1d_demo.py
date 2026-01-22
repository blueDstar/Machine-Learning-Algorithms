import numpy as np
import math
n = np.array([15,15,16,19,19,20,20,21,22,28,35,40,41,42,43,44,60,61,65,])
K=2
centroids = np.array([16,22], dtype='float32')
times = 1
for _ in range(times):  
    distances = np.array([[abs(x - c) for c in centroids] for x in n])
    
    clusters = np.argmin(distances, axis=1)
    
    new_centroids = np.array([np.mean(n[clusters == i]) if np.any(clusters == i) else centroids[i] 
                              for i in range(K)])

    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids
print("khoảng cách đo", distances )
# print(f"Tâm cụm cuối cùng: {centroids}")
# print(f"Phân cụm dữ liệu: {clusters}")
