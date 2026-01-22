import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread(r"F:\Githubrestore\Segmentation-main\Segmentation-main\lena.tif")
ro, co, ch = img.shape

kernel = np.array([[1, 1],
                   [1, 1]])

new_img_0 = np.zeros((ro // 2, co // 2), dtype=np.uint8)
new_img_1 = np.zeros((ro // 2, co // 2), dtype=np.uint8)
new_img_2 = np.zeros((ro // 2, co // 2), dtype=np.uint8)

def process(new_img, channel):
    for i in range(0, ro, 2):
        for j in range(0, co, 2):
            if i + 1 < ro and j + 1 < co:
                block = img[i:i + 2, j:j + 2, channel]
                mul = kernel * block
                new_point = np.round(np.mean(mul))
                new_img[i // 2, j // 2] = new_point
    return new_img

a = process(new_img_0, 0)
b = process(new_img_1, 1)
c = process(new_img_2, 2)

new_img = cv2.merge((a, b, c))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(new_img)
plt.title("Downsampled (2x2 mean)")
plt.axis("off")

out_path = os.path.join(os.path.dirname(__file__), "downsample_2x2_mean_result.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
