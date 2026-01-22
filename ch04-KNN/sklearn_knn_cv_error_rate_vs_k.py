import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([
    [40, 20], [50, 50], [60, 90], [10, 25], [70, 70],
    [60, 10], [25, 80], [15, 10], [55, 65], [35, 45],
    [20, 40], [80, 90], [75, 65], [65, 85], [45, 35]
], dtype="float32")

y_train = np.array([
    "Red", "Blue", "Blue", "Red", "Blue",
    "Red", "Blue", "Red", "Blue", "Red",
    "Red", "Blue", "Blue", "Blue", "Red"
])

k_range = range(1, 11)
error_rate = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=6, scoring="accuracy")
    error_rate.append(1 - scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rate, linestyle="dashed", marker="o")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.title("Error Rate vs K")

out_path = os.path.join(os.path.dirname(__file__), "knn_error_rate_vs_k.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()

optimal_k = list(k_range)[int(np.argmin(error_rate))]
print(f"K tối ưu: {optimal_k}")
