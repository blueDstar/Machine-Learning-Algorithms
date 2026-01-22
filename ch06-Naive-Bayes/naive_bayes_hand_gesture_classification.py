import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_dir = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\Chap 6 Hand\small_hand_gesture"
class_dirs = [os.path.join(data_dir, f"class_{i}") for i in range(6)]
class_names = [f"class_{i}" for i in range(6)]

IMG_SIZE = 100
images = []
labels = []

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten() / 255.0
    return img

for label, img_dir in enumerate(class_dirs):
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = preprocess_image(img_path)
        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

plt.figure(figsize=(8, 5))
sns.countplot(x=labels, palette="viridis")
plt.title("Phân bố số lượng ảnh theo từng lớp")
plt.xlabel("Lớp")
plt.ylabel("Số lượng ảnh")
plt.xticks(ticks=range(6), labels=class_names)
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=[f'class_{i}' for i in range(6)],
            yticklabels=[f'class_{i}' for i in range(6)])
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Actual', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.savefig('confusion_matrix.png')
plt.show()

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.iloc[:-1, :-1].plot(kind="bar", figsize=(10, 5))
plt.title("So sánh Precision, Recall, F1-Score giữa các lớp")
plt.xticks(rotation=45)
plt.show()

correct = np.sum(y_pred == y_test)
incorrect = np.sum(y_pred != y_test)
plt.figure(figsize=(6, 6))
plt.pie([correct, incorrect], labels=["Đúng", "Sai"], autopct="%1.1f%%", colors=["green", "red"], startangle=90)
plt.title("Tỉ lệ dự đoán đúng & sai")
plt.show()

def display_predictions(X_test, y_test, y_pred, num_images=12):
    plt.figure(figsize=(15, 22))
    rows = num_images // 4 if num_images % 4 == 0 else (num_images // 4) + 1
    
    for i in range(min(num_images, len(X_test))):
        plt.subplot(rows, 4, i+1)
        img = X_test[i][:IMG_SIZE*IMG_SIZE].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(img, cmap='gray')
        color = 'green' if y_test[i] == y_pred[i] else 'red'
        plt.title(f'Thực: {y_test[i]}\nDự đoán: {y_pred[i]}', color=color)
        plt.axis('off')
    plt.suptitle('Một số dự đoán mẫu (Xanh: Đúng, Đỏ: Sai)', fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

display_predictions(X_test, y_test, y_pred, num_images=24)

accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình Naive Bayes: {accuracy * 100:.2f}%")
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=class_names))
