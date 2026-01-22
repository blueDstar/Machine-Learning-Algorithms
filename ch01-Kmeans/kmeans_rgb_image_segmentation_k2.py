import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

img = cv2.imread(r"F:\StudyatCLass\Study\class\Thigiacmaytinh\baitap\lena.tif")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = img_rgb.reshape((-1,3))
k = 2
kmeans = KMeans(n_clusters=k, random_state =42)
kmeans.fit(pixels)
labels = kmeans.labels_
segmented_img = kmeans.cluster_centers_.astype('uint8')[labels]
segmented_img = segmented_img.reshape(img_rgb.shape)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Ảnh gốc")
plt.imshow(img_rgb)
plt.axis('off')
plt.subplot(1,2,2)
plt.title("Ảnh phân đoạn Kmeans")
plt.imshow(segmented_img)
plt.axis('off')

plt.show()