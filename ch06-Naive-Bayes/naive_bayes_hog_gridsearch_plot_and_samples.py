import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from skimage import exposure

base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\Chap 6 Hand\small_hand_gesture"
class_dirs = [os.path.join(data_dir, f"class_{i}") for i in range(6)]
class_names = [f"class_{i}" for i in range(6)]

IMG_SIZE = 128
images = []
labels = []

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

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

for label, img_dir in enumerate(class_dirs):
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        try:
            images.append(extract_features(img_path))
            labels.append(label)
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {e}")

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

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

print(f"Tham số tối ưu: {grid_search.best_params_}")
print(f"Độ chính xác tốt nhất trên tập validation: {grid_search.best_score_:.4f}")

best_nb = grid_search.best_estimator_
best_nb.fit(X_train, y_train)

y_pred = best_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nĐộ chính xác trên tập test: {accuracy * 100:.2f}%")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=class_names))

plt.figure(figsize=(10, 6))
plt.semilogx(
    param_grid["var_smoothing"],
    grid_search.cv_results_["mean_test_score"],
    marker="o"
)
plt.xlabel("var_smoothing (log scale)")
plt.ylabel("Accuracy")
plt.title("Grid Search Performance vs var_smoothing")
plt.grid(True)

out_path = os.path.join(base_dir, "var_smoothing_gridsearch_curve.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()

def display_predictions(X_test, y_test, y_pred, num_images=12):
    n = min(num_images, len(X_test))
    plt.figure(figsize=(15, 10))
    cols = 4
    rows = n // cols + (1 if n % cols else 0)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img_flat = X_test[i][:IMG_SIZE * IMG_SIZE]
        img = img_flat.reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap="gray")
        plt.title(f"T:{y_test[i]} P:{y_pred[i]}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(base_dir, "sample_predictions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)
    plt.show()

display_predictions(X_test, y_test, y_pred, num_images=12)
