import os
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

img_path = r"F:\Githubrestore\Segmentation-main\Segmentation-main\lena.tif"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

img_small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
X = img_small.reshape(-1, 3).astype(np.float32)

k = 4
model = AgglomerativeClustering(n_clusters=k, linkage="ward")
labels = model.fit_predict(X)

centers = np.zeros((k, 3), dtype=np.float32)
for i in range(k):
    mask = labels == i
    centers[i] = X[mask].mean(axis=0) if np.any(mask) else 0

seg = centers[labels].reshape(img_small.shape).astype(np.uint8)

side_by_side = np.hstack([img_small, seg])
cv2.imshow("Original | Agglomerative Segmentation", side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()

out_dir = r"F:\Githubrestore\Segmentation-main\Segmentation-main\chap3"
os.makedirs(out_dir, exist_ok=True)

out_path_seg = os.path.join(out_dir, f"agglomerative_seg_k{k}_128x128.png")
out_path_side = os.path.join(out_dir, f"agglomerative_side_by_side_k{k}_128x128.png")

cv2.imwrite(out_path_seg, seg)
cv2.imwrite(out_path_side, side_by_side)

print("Saved:", out_path_seg)
print("Saved:", out_path_side)
