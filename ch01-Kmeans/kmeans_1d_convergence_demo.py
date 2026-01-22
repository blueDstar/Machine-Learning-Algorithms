import numpy as np
n = np.array([15,15,16,19,19,20,20,21,22,28,35,40,41,42,43,44,60,61,65,])
K=2
centroids = np.array([16,22], dtype='float32')
times = 8
for i in range(times):
    print(f"\n Lần lặp {i+1}")
    clusters = [[]for _ in range(K)]
    d = []
    for x in n:
        distances = [abs(x-c) for c in centroids]
        closest_cluster = np.argmin(distances)
        clusters[closest_cluster].append(x)
        d.append(distances)
    print(f"khoảng cách: {d}")
    for j in range(K):
        print(f"Nhóm {j+1}: {clusters[j]}")
    prev_centroids = centroids.copy()
    for j in range (K):
        if len(clusters[j]) > 0:
            centroids[j] = np.mean(clusters[j])
    print(f"tâm mới: {centroids}")
    if np.array_equal(centroids, prev_centroids):
        print(f"Thuật toán hội tụ sau {i+1} lần lặp.")
        break
