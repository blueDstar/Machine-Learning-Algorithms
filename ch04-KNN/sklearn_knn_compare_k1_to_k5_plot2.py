import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([
    [167, 51], [182, 62], [176, 69], [173, 64], [172, 65],
    [174, 56], [169, 58], [173, 57], [170, 55]
])
y_train = np.array([
    "Underweight", "Normal", "Normal", "Normal", "Normal",
    "Underweight", "Normal", "Normal", "Normal"
])

X_test = np.array([[170, 57]])

predicted_labels = []
for k in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predicted_labels.append(knn.predict(X_test)[0])

under_idx = np.where(y_train == "Underweight")[0]
norm_idx = np.where(y_train == "Normal")[0]

plt.figure(figsize=(12, 6))

for i, k in enumerate(range(1, 6)):
    plt.subplot(2, 3, i + 1)
    plt.title(f"K={k}: {predicted_labels[i]}", fontsize=12, fontweight="bold")

    plt.scatter(X_train[under_idx, 0], X_train[under_idx, 1], label="Underweight", marker="o")
    plt.scatter(X_train[norm_idx, 0], X_train[norm_idx, 1], label="Normal", marker="s")

    plt.scatter(X_test[0, 0], X_test[0, 1], label="Test Point", marker="x", s=100)
    plt.text(X_test[0, 0], X_test[0, 1] + 2, "Target Point", fontsize=10)

    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, linestyle="-", alpha=0.5)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "sklearn_knn_compare_k1_to_k5_2.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
