import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

n = np.array([[1,1], [3,2], [9,1], [3,7], [7,2], [9,7], [4,8], [8,3], [1,4]])

Z1 = sch.linkage(n, method='single')

f1 = fcluster(Z1, 4, criterion='maxclust')

print(f"Clusters: {f1}")

plt.figure(figsize=(8, 5))
sch.dendrogram(Z1, labels=[f"p{i+1}" for i in range(len(n))])
plt.title("Dendrogram - Single Linkage")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()
