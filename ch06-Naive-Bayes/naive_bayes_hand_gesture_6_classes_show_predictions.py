import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\Chap 6 Hand\small_hand_gesture"
class_dirs = [os.path.join(data_dir, f"class_{i}") for i in range(6)]
class_names = [f"class_{i}" for i in range(6)]

IMG_SIZE = 128
images = []
labels = []

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten() / 255.0
    return img

for label, img_dir in enumerate(class_dirs):
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        images.append(preprocess_image(img_path))
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

def display_predictions(images, labels, predictions, num_images=24):
    n = min(num_images, len(images))
    cols = 6
    rows = n // cols + (1 if n % cols else 0)

    plt.figure(figsize=(15, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = images[i].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap="gray")
        plt.title(f"Thực: {labels[i]} | Dự đoán: {predictions[i]}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(base_dir, "sample_predictions_6_classes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)
    plt.show()

display_predictions(X_test, y_test, y_pred, num_images=24)

accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình Naive Bayes: {accuracy * 100:.2f}%")
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=class_names))
