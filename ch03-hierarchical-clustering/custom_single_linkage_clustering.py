import numpy as np
from scipy.cluster.hierarchy import fcluster

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

def compute_distance_matrix(X):
    n = len(X)
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = euclidean_distance(X[i], X[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

def find_closest_cluster(dist_matrix, clusters):
    min_dist = np.inf
    pair = (-1, -1)
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            for point1 in clusters[i]:
                for point2 in clusters[j]:
                    dist = dist_matrix[point1, point2]
                    if dist < min_dist:
                        min_dist = dist
                        pair = (i,j)
    return pair, min_dist

def single_linkage_clustering(X):
    n = len(X)
    dist_matrix = compute_distance_matrix(X)
    clusters = [[i] for i in range(n)]
    Z = []
    cluster_counter = n
    cluster_map = {i: i for i in range(n)}

    while len(clusters) > 1:
        (i,j), min_dist = find_closest_cluster(dist_matrix, clusters)
        idx1 = cluster_map[clusters[i][0]]
        idx2 = cluster_map[clusters[j][0]]

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            

