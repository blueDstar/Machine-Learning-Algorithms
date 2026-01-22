import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = r"F:\Githubrestore\Segmentation-main\Segmentation-main\lena.tif"
anh = plt.imread(img_path)
print("Original shape:", anh.shape)

H, W = anh.shape[:2]

desired_w = int(735 / 2)
desired_h = int(1307 / 2)

scale = min(W / desired_w, H / desired_h, 1.0)
new_w = max(1, int(desired_w * scale))
new_h = max(1, int(desired_h * scale))

new_dimensions = cv2.resize(anh, (new_w, new_h))
print("Resized shape:", new_dimensions.shape)

nenden = np.zeros_like(anh)
nenden1 = np.zeros_like(anh)
nenden2 = np.zeros_like(anh)
nenden3 = np.zeros_like(anh)
nenden4 = np.zeros_like(anh)

h, w = new_dimensions.shape[:2]

nenden[:h, :w] = new_dimensions
nenden1[-h:, :w] = new_dimensions
nenden2[:h, -w:] = new_dimensions
nenden3[-h:, -w:] = new_dimensions

nenden4[:h, :w] = new_dimensions
nenden4[-h:, :w] = new_dimensions
nenden4[:h, -w:] = new_dimensions
nenden4[-h:, -w:] = new_dimensions

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(nenden3)
plt.title("1")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(nenden)
plt.title("2")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(nenden1)
plt.title("3")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(nenden2)
plt.title("4")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(nenden4)
plt.title("5")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(anh)
plt.title("Original")
plt.axis("off")

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "image_quadrant_paste_collage.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
