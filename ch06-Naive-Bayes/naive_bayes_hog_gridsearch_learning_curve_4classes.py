import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage import exposure
import seaborn as sns
from time import time

plt.style.use("seaborn-v0_8")
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\Chap 6 Hand\small_hand_gesture"
class_names = [f"class_{i}" for i in range(4)]
class_dirs = [os.path.join(data_dir, name) for name in class_names]

IMG_SIZE = 128
images = []
labels = []

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {img_path}")

    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    fd, hog_image = hog(
        img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    features = np.concatenate([
        img.flatten() / 255.0,
        hog_image.flatten(),
        fd
    ])
    return features

print("Đang tải và xử lý dữ liệu...")
start_time = time()
for label, img_dir in enumerate(class_dirs):
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        try:
            images.append(extract_features(img_path))
            labels.append(label)
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {str(e)}")

images = np.array(images)
labels = np.array(labels)

print(f"Hoàn thành trong {time() - start_time:.2f} giây")
print(f"Tổng số mẫu: {len(images)}")

plt.figure(figsize=(10, 6))
sns.countplot(x=labels)
plt.xticks(ticks=range(len(class_names)), labels=class_names)
plt.title("Phân bố dữ liệu theo lớp")
plt.xlabel("Lớp")
plt.ylabel("Số lượng mẫu")
out_path = os.path.join(base_dir, "class_distribution.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print("\nĐang tối ưu siêu tham số...")
param_grid = {"var_smoothing": np.logspace(-12, -2, 50)}

grid_search = GridSearchCV(
    GaussianNB(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nTham số tối ưu: {grid_search.best_params_}")
print(f"Độ chính xác tốt nhất: {grid_search.best_score_:.4f}")

plt.figure(figsize=(12, 6))
plt.semilogx(
    param_grid["var_smoothing"],
    grid_search.cv_results_["mean_test_score"],
    marker="o",
    linewidth=2
)
plt.fill_between(
    param_grid["var_smoothing"],
    grid_search.cv_results_["mean_test_score"] - grid_search.cv_results_["std_test_score"],
    grid_search.cv_results_["mean_test_score"] + grid_search.cv_results_["std_test_score"],
    alpha=0.2
)
plt.xlabel("Giá trị var_smoothing (log scale)")
plt.ylabel("Độ chính xác")
plt.title("Hiệu suất theo siêu tham số")
plt.grid(True, which="both", ls="--")
out_path = os.path.join(base_dir, "hyperparameter_performance.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.close()

best_nb = grid_search.best_estimator_

train_sizes, train_scores, test_scores = learning_curve(
    best_nb,
    X_train,
    y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy"
)

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training score")
plt.plot(train_sizes, np.mean(test_scores, axis=1), "o-", label="Validation score")
plt.fill_between(
    train_sizes,
    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
    alpha=0.1
)
plt.fill_between(
    train_sizes,
    np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
    np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
    alpha=0.1
)
plt.xlabel("Số lượng mẫu huấn luyện")
plt.ylabel("Độ chính xác")
plt.title("Đường cong học tập")
plt.legend(loc="best")
plt.grid(True)
out_path = os.path.join(base_dir, "learning_curve.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.close()

best_nb.fit(X_train, y_train)
y_pred = best_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nĐộ chính xác trên tập test: {accuracy * 100:.2f}%")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Ma trận nhầm lẫn")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
out_path = os.path.join(base_dir, "confusion_matrix.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.close()

def display_predictions(X_test, y_test, y_pred, num_images=12):
    n = min(num_images, len(X_test))
    cols = 4
    rows = n // cols + (1 if n % cols else 0)

    plt.figure(figsize=(15, 10))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = X_test[i][:IMG_SIZE * IMG_SIZE].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap="gray")
        color = "green" if y_test[i] == y_pred[i] else "red"
        plt.title(f"Thực: {y_test[i]}\nDự đoán: {y_pred[i]}", color=color, fontsize=10)
        plt.axis("off")

    plt.suptitle("Một số dự đoán mẫu (Xanh: Đúng, Đỏ: Sai)")
    plt.tight_layout()
    out_path = os.path.join(base_dir, "sample_predictions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)
    plt.close()

display_predictions(X_test, y_test, y_pred)

print("\nĐã hoàn thành quá trình huấn luyện và đánh giá!")
print("Các biểu đồ đã được lưu vào thư mục chứa file .py.")
