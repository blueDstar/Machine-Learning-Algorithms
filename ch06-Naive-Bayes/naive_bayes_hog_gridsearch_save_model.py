import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\Chap 6 Hand\small_hand_gesture"
class_dirs = [os.path.join(data_dir, f"class_{i}") for i in range(6)]
class_names = [f"class_{i}" for i in range(6)]

IMG_SIZE = 128
features_list = []
labels = []
img_paths = []

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    features, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    return features

for label, img_dir in enumerate(class_dirs):
    for img_name in os.listdir(img_dir):
        path = os.path.join(img_dir, img_name)
        feat = preprocess_image(path)
        features_list.append(feat)
        labels.append(label)
        img_paths.append(path)

X = np.array(features_list)
y = np.array(labels)
img_paths = np.array(img_paths)

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, img_paths, test_size=0.2, random_state=42, stratify=y
)

param_grid = {"var_smoothing": np.logspace(-9, 0, 10)}
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_nb_model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")

best_nb_model.fit(X_train, y_train)
y_pred = best_nb_model.predict(X_test)

plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette="viridis")
plt.title("Phân bố số lượng ảnh theo từng lớp")
plt.xlabel("Lớp")
plt.ylabel("Số lượng ảnh")
plt.xticks(ticks=range(6), labels=class_names)
out_path = os.path.join(base_dir, "class_distribution.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Thực tế", fontsize=12)
plt.ylabel("Dự đoán", fontsize=12)
out_path = os.path.join(base_dir, "confusion_matrix.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names)
df_report = pd.DataFrame(report).transpose()
df_report.iloc[:-1, :-1].plot(kind="bar", figsize=(10, 5))
plt.title("So sánh Precision, Recall, F1-Score giữa các lớp")
plt.xticks(rotation=45)
plt.tight_layout()
out_path = os.path.join(base_dir, "classification_report_bar.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

correct = int(np.sum(y_pred == y_test))
incorrect = int(np.sum(y_pred != y_test))
plt.figure(figsize=(6, 6))
plt.pie([correct, incorrect], labels=["Đúng", "Sai"], autopct="%1.1f%%", colors=["green", "red"], startangle=90)
plt.title("Tỉ lệ dự đoán đúng & sai")
out_path = os.path.join(base_dir, "accuracy_pie.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

def display_predictions(paths, y_true, y_hat, num_images=24):
    plt.figure(figsize=(15, 10))
    cols = 6
    rows = num_images // cols + (1 if num_images % cols else 0)

    for i in range(min(num_images, len(paths))):
        img = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        color = "green" if y_true[i] == y_hat[i] else "red"
        plt.title(f"T:{y_true[i]} P:{y_hat[i]}", color=color, fontsize=10)
        plt.axis("off")

    plt.suptitle("Một số dự đoán mẫu (Xanh: Đúng, Đỏ: Sai)", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(base_dir, "sample_predictions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)
    plt.show()

display_predictions(paths_test, y_test, y_pred, num_images=24)

model_path = os.path.join(base_dir, "best_naive_bayes_model.pkl")
joblib.dump(best_nb_model, model_path)
print("Saved model:", model_path)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=class_names))
