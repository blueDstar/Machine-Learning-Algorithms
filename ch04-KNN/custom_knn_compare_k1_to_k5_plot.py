import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classify(X_train, y_train, X_test, k=5):
    distances = [(euclidean_distance(X_test, x), label) for x, label in zip(X_train, y_train)]
    k_neighbors = sorted(distances, key=lambda t: t[0])[:k]
    labels = [label for _, label in k_neighbors]
    return Counter(labels).most_common(1)[0][0]

X_train = np.array([
    [167, 51], [182, 62], [176, 69], [173, 64], [172, 65],
    [174, 56], [169, 58], [173, 57], [170, 55]
])
y_train = np.array([
    "Underweight", "Normal", "Normal", "Normal", "Normal",
    "Underweight", "Normal", "Normal", "Normal"
])

X_test = np.array([170, 57])

predicted_labels = [knn_classify(X_train, y_train, X_test, k) for k in range(1, 6)]

X_under = X_train[y_train == "Underweight"]
X_norm = X_train[y_train == "Normal"]

plt.figure(figsize=(12, 6))

for i, k in enumerate(range(1, 6)):
    plt.subplot(2, 3, i + 1)
    plt.title(f"K={k}: {predicted_labels[i]}", fontsize=12, fontweight="bold")

    plt.scatter(X_under[:, 0], X_under[:, 1], label="Underweight", marker="o")
    plt.scatter(X_norm[:, 0], X_norm[:, 1], label="Normal", marker="s")
    plt.scatter(X_test[0], X_test[1], label="Test Point", marker="x", s=100)

    plt.text(X_test[0], X_test[1] + 2, "Target Point", fontsize=10)

    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, linestyle="-", alpha=0.5)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "custom_knn_compare_k1_to_k5_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
