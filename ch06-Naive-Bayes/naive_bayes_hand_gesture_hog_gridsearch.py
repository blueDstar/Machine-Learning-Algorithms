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
class_dirs = [os.path.join(data_dir, f"class_{i}") for i in range(6)]

IMG_SIZE = 150
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
        orientations=10,
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
            features = extract_features(img_path)
            images.append(features)
            labels.append(label)
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {str(e)}")

images = np.array(images)
labels = np.array(labels)

print(f"Hoàn thành trong {time() - start_time:.2f} giây")
print(f"Tổng số mẫu: {len(images)}")

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

plt.figure(figsize=(10, 6))
sns.countplot(x=labels)
plt.title("Phân bố các lớp trong tập dữ liệu")
plt.xlabel("Lớp")
plt.ylabel("Số lượng mẫu")
out_path = os.path.join(base_dir, "class_distribution.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

print("\nĐang tìm kiếm siêu tham số tối ưu...")
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

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"\nTham số tối ưu: {best_params}")
print(f"Độ chính xác tốt nhất trên tập validation: {best_score:.4f}")

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
plt.xlabel("Giá trị var_smoothing (log scale)", fontsize=12)
plt.ylabel("Độ chính xác", fontsize=12)
plt.title("Hiệu suất theo siêu tham số với Cross-Validation", fontsize=14)
plt.grid(True, which="both", ls="--")
out_path = os.path.join(base_dir, "hyperparameter_performance.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

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
plt.plot(train_sizes, np.mean(test_scores, axis=1), "o-", label="Cross-validation score")
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
plt.xlabel("Số lượng mẫu huấn luyện", fontsize=12)
plt.ylabel("Độ chính xác", fontsize=12)
plt.title("Learning Curve", fontsize=14)
plt.legend(loc="best")
plt.grid(True)
out_path = os.path.join(base_dir, "learning_curve.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

best_nb.fit(X_train, y_train)
y_pred = best_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nĐộ chính xác trên tập test: {accuracy * 100:.2f}%")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=[f"class_{i}" for i in range(6)]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"class_{i}" for i in range(6)],
    yticklabels=[f"class_{i}" for i in range(6)]
)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Dự đoán", fontsize=12)
plt.ylabel("Thực tế", fontsize=12)
out_path = os.path.join(base_dir, "confusion_matrix.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

means = best_nb.theta_
feature_importance = np.mean(np.abs(means[1:] - means[:-1]), axis=0)

top_indices = np.argsort(feature_importance)[-20:][::-1]
top_importance = feature_importance[top_indices]

plt.figure(figsize=(12, 6))
plt.bar(range(len(top_importance)), top_importance)
plt.xticks(range(len(top_importance)), top_indices, rotation=45)
plt.xlabel("Feature Index", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.title("Top 20 Important Features", fontsize=14)
plt.grid(True, axis="y")
out_path = os.path.join(base_dir, "feature_importance.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

def display_predictions(X_test, y_test, y_pred, num_images=12):
    plt.figure(figsize=(15, 10))
    rows = num_images // 4 if num_images % 4 == 0 else (num_images // 4) + 1

    for i in range(min(num_images, len(X_test))):
        plt.subplot(rows, 4, i + 1)
        img = X_test[i][:IMG_SIZE * IMG_SIZE].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap="gray")
        color = "green" if y_test[i] == y_pred[i] else "red"
        plt.title(f"Thực: {y_test[i]}\nDự đoán: {y_pred[i]}", color=color)
        plt.axis("off")

    plt.suptitle("Một số dự đoán mẫu (Xanh: Đúng, Đỏ: Sai)", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(base_dir, "sample_predictions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)
    plt.show()

display_predictions(X_test, y_test, y_pred)
