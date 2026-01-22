import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

X_train = np.array([[40,20], [50,50], [60,90], [10,25], [70,70], [60,10], [25,80]])
y_train = np.array(["red", "blue", "blue", "red", "blue", "red", "blue"])

X_test = np.array([20, 35])

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classify(X_train, y_train, X_test, k=5):
    distances = [(euclidean_distance(X_test, x), label) for x, label in zip(X_train, y_train)]
    k_neighbors = sorted(distances)[:k]
    labels = [label for _, label in k_neighbors]
    return Counter(labels).most_common(1)[0][0]

predicted_label = knn_classify(X_train, y_train, X_test, k=3)
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label}")

X_red = X_train[y_train == "red"]
X_blue = X_train[y_train == "blue"]

plt.figure(figsize=(10,6))
plt.scatter(X_red[:,0], X_red[:,1], color='red', label="Red")
plt.scatter(X_blue[:,0], X_blue[:,1], color='blue', label="Blue", marker='s')
plt.scatter(X_test[0], X_test[1], color='purple', label="Test Point [20,35]", marker='x', s=100)

plt.text(X_test[0], X_test[1] + 3, "Target Point", fontsize=12, color='red')

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title(f"Predict Class {X_test} with K=3: {predicted_label}")
plt.legend()
plt.grid(True, linestyle="-", alpha=0.5)

out_path = os.path.join(os.path.dirname(__file__), "knn_predict_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
