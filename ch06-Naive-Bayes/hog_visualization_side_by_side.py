import os
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

img_path = r"C:\Users\ACER\OneDrive\Pictures\Desktop\481769936_574471732280855_3957250888316744881_n.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

fd, hog_image = hog(
    img,
    orientations=10,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Ảnh gốc")
ax[0].axis("off")

ax[1].imshow(hog_image, cmap="gray")
ax[1].set_title("Ảnh HOG")
ax[1].axis("off")

out_path = os.path.join(os.path.dirname(__file__), "hog_visualization.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
