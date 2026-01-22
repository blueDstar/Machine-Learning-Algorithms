import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Dữ liệu
n = np.array([[1,1], [3,2], [9,1], [3,7], [7,2], [9,7], [4,8], [8,3], [1,4]])
labels = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]

# Các phương pháp liên kết
methods = ["Single", "Complete", "Centroid", "Average", "Ward"]
linkage_matrices = [
    sch.linkage(n, method='single'),
    sch.linkage(n, method='complete'),
    sch.linkage(n, method='centroid'),
    sch.linkage(n, method='average'),
    sch.linkage(n, method='ward')
]

# Vẽ biểu đồ
plt.figure(figsize=(12,6))

for i in range(len(methods)):
    plt.subplot(2, 3, i + 1)  # 2 hàng, tối đa 3 cột
    plt.title(methods[i], fontsize=14, fontweight="bold")
    sch.dendrogram(linkage_matrices[i], labels=labels, leaf_font_size=10)
    plt.xlabel("Points", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
plt.tight_layout()
plt.show()
