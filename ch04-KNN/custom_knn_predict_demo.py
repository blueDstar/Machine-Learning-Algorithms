import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classify(X_train, y_train, X_test, k=5):
    distances = [(euclidean_distance(X_test, x), label) for x, label in zip(X_train, y_train)]
    k_neighbors = sorted(distances)[:k]
    labels = [label for _, label in k_neighbors]
    return Counter(labels).most_common(1)[0][0]

X_train = np.array([[40,20], [50,50], [60,90], [10,25], [70,70], [60,10], [25,80]])
y_train = np.array(["red", "blue", "blue", "red" , "blue", "red" , "blue"])

X_test = np.array([20, 35])

predicted_label = knn_classify(X_train, y_train, X_test, k=3)
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label}")
