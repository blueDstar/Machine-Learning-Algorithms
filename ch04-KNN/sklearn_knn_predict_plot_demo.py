import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([[40, 20], [50, 50], [60, 90], [10, 25], [70, 70], [60, 10], [25, 80]])
y_train = np.array(["red", "blue", "blue", "red", "blue", "red", "blue"])

X_test = np.array([[20, 35]])

predicted_labels = []
for k in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predicted_labels.append(knn.predict(X_test)[0])

red_idx = np.where(y_train == "red")[0]
blue_idx = np.where(y_train == "blue")[0]

plt.figure(figsize=(10, 6))
plt.scatter(X_train[red_idx, 0], X_train[red_idx, 1], label="Red", edgecolors="black")
plt.scatter(X_train[blue_idx, 0], X_train[blue_idx, 1], label="Blue", marker="s", edgecolors="black")
plt.scatter(X_test[0, 0], X_test[0, 1], label="Test Point [20,35]", marker="x", s=100, linewidths=2)

plt.text(X_test[0, 0] + 2, X_test[0, 1] + 2, "Target Point", fontsize=12)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title(f"Predict Class {X_test[0]} with K=5: {predicted_labels[-1]}")
plt.legend()
plt.grid(True, linestyle="-", alpha=0.5)

out_path = os.path.join(os.path.dirname(__file__), "sklearn_knn_predict_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
