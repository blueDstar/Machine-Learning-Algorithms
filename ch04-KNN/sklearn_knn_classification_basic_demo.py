import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([[40,20], [50,50], [60,90], [10,25], [70,70], [60,10], [25,80]], dtype='float32')
y_train = np.array(["red", "blue", "blue", "red" , "blue", "red" , "blue"])

knn_regressor = KNeighborsClassifier(n_neighbors=3)
knn_regressor.fit(X_train, y_train)

X_test = np.array([[20, 35]])

predicted_label = knn_regressor.predict(X_test)
print(f"Nhãn dự đoán cho điểm {X_test} là: {predicted_label}")

