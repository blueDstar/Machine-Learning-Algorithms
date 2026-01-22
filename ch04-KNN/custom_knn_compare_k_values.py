import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classify(X_train, y_train, X_test, k=5):
    distances = [(euclidean_distance(X_test, x), label) for x, label in zip(X_train, y_train)]
    k_neighbors = sorted(distances)[:k]
    labels = [label for _, label in k_neighbors]
    return Counter(labels).most_common(1)[0][0]

X_train = np.array([[167,51], [182,62], [176,69], [173,64], [172,65], [174,56], [169,58], [173,57], [170,55]])
y_train = np.array(["Underweight", "Normal", "Normal", "Normal" , "Normal", "Underweight" , "Normal", "Normal", "Normal"])

X_test = np.array([170, 57])

predicted_label1 = knn_classify(X_train, y_train, X_test, k=1)
predicted_label2 = knn_classify(X_train, y_train, X_test, k=2)
predicted_label3 = knn_classify(X_train, y_train, X_test, k=3)
predicted_label4 = knn_classify(X_train, y_train, X_test, k=4)
predicted_label5 = knn_classify(X_train, y_train, X_test, k=5)
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label1}")
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label2}")
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label3}")
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label4}")
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label5}")