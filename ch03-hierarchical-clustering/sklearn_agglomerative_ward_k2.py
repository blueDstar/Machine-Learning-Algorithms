import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

X1 = np.array([[1,1], [3,2], [9,1], [3,7], [7,2], [9,7], [4,8], [8,3], [1,4]])
Z1 = AgglomerativeClustering(n_clusters =2, linkage = 'ward')
Z1.fit_predict(X1)
print(Z1.labels_) 
fig, ax = plt.subplots(figsize=(10,6))
scatter = ax.scatter(X1[:,0], X1[:,1], c=Z1.labels_, cmap='rainbow')
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 11, 1))
plt.grid(True, linestyle="-", alpha=0.5)
legend = ax.legend(*scatter.legend_elements(), title ='Nhóm', bbox_to_anchor=(1,1))
ax.add_artist(legend)
plt.title('Phân nhóm phân cấp')
plt.show()   