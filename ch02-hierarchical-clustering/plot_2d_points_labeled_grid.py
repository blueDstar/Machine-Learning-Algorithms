import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu ban đầu
n = np.array([[1,1], [3,2], [9,1], [3,7], [7,2], [9,7], [4,8], [8,3], [1,4]])
num_points = len(n)

# # Tính ma trận khoảng cách ban đầu
# distance_matrix = np.full((num_points, num_points), np.inf)
# for i in range(num_points):
#     for j in range(i+1, num_points):
#         distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(n[i] - n[j])

# # Danh sách cụm ban đầu (mỗi điểm là một cụm riêng)
# clusters = {i: [i] for i in range(num_points)}

# # Tính tổng phương sai trong một cụm
# def compute_variance(cluster):
#     points = np.array([n[i] for i in cluster])
#     centroid = np.mean(points, axis=0)
#     variance = np.sum((points - centroid) ** 2)
#     return variance

# # Tìm hai cụm có tổng phương sai nhỏ nhất sau khi gộp
# def find_best_merge():
#     min_increase = np.inf
#     best_pair = (-1, -1)
    
#     for i in clusters:
#         for j in clusters:
#             if i >= j:
#                 continue
#             merged_cluster = clusters[i] + clusters[j]
#             new_variance = compute_variance(merged_cluster)
#             variance_increase = new_variance - (compute_variance(clusters[i]) + compute_variance(clusters[j]))
            
#             if variance_increase < min_increase:
#                 min_increase = variance_increase
#                 best_pair = (i, j)
    
#     return best_pair

# # Thuật toán phân cụm
# while len(clusters) > 1:
#     c1, c2 = find_best_merge()
    
#     # Gộp hai cụm
#     clusters[c1] += clusters[c2]
#     del clusters[c2]

#     # Cập nhật khoảng cách giữa cụm mới và các cụm còn lại
#     for i in clusters:
#         if i != c1:
#             distance_matrix[c1, i] = distance_matrix[i, c1] = np.linalg.norm(
#                 np.mean([n[k] for k in clusters[c1]], axis=0) - np.mean([n[k] for k in clusters[i]], axis=0)
#             )

#     # Xóa dòng/cột của cụm cũ khỏi ma trận khoảng cách
#     distance_matrix[:, c2] = np.inf
#     distance_matrix[c2, :] = np.inf
    
#     print(f"Gộp cụm {c1} và {c2} -> {clusters[c1]}")

# # Kết quả cuối cùng
# print("\nCụm cuối cùng:", list(clusters.values())[0])

# Vẽ dữ liệu trên lưới
plt.figure(figsize=(10,6))
plt.scatter(n[:,0], n[:,1], color='red', label="Data Points")
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 11, 1))
plt.grid(True, linestyle="-", alpha=0.5)

# Gắn nhãn cho các điểm
for i, (x, y) in enumerate(n):
    plt.text(x + 0.1, y+0.1, f"p{i+1}", fontsize=12, color='black')

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Data Points on Grid")
plt.legend()
plt.show()