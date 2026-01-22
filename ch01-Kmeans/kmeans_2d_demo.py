import numpy as np
data = np.array([[2,10], [2,5], [8,4], [5,8], [7,5], [6,4], [1,2], [4,9]],dtype = 'float32')
# data = np.array([[2,10], [2,5], [8,4], [5,8], [7,5], [6,4], [1,2], [4,9]]).reshape(-1,1)
K = 3
centroids = np.array([[2,10], [5,8], [1,2]], dtype='float32')
times = 100
clusters = [[] for _ in range(K)]
prev_centroids = np.zeros_like(centroids, dtype = 'float32')

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
print(f"matran 2 chieu {data}")
print(f"kiểu dữ liệu {data.dtype}")