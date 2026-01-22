import os
import cv2
import matplotlib.pyplot as plt

img_path = r"F:\Githubrestore\Segmentation-main\Segmentation-main\lena.tif"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)

image_with_keypoints = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(8, 6))
plt.imshow(image_with_keypoints, cmap="gray")
plt.title("SIFT Keypoints")
plt.axis("off")

out_path = os.path.join(os.path.dirname(__file__), "sift_keypoints.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
