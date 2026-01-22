import os
import cv2
import numpy as np

img_path = r"F:\Githubrestore\Segmentation-main\Segmentation-main\lena.tif"
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

step = 10
size = 20

def computer_gradients(gray):
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return sobel_x, sobel_y

def compute_magnitude_and_orientation(grad_x, grad_y):
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    orientation = (np.arctan2(grad_y, grad_x) * (180.0 / np.pi)) % 360.0
    return magnitude, orientation

def compute_dense_sift(gray, step, size):
    descriptors = []
    keypoints = []
    h, w = gray.shape[:2]

    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            region = gray[y:y + size, x:x + size]
            grad_x, grad_y = computer_gradients(region)
            mag, ori = compute_magnitude_and_orientation(grad_x, grad_y)

            hist, _ = np.histogram(
                ori.flatten(),
                bins=8,
                range=(0, 360),
                weights=mag.flatten()
            )
            s = np.sum(hist)
            hist = hist / s if s > 0 else hist

            descriptors.append(hist)
            keypoints.append((x + size // 2, y + size // 2))

    return np.array(descriptors), keypoints

descriptors_manual, keypoints_manual = compute_dense_sift(gray_image, step, size)
print("số lượng đặc trưng:", len(descriptors_manual))

kps = [cv2.KeyPoint(float(x), float(y), float(size)) for x, y in keypoints_manual]
image_manual = cv2.drawKeypoints(gray_image, kps, None)

cv2.imshow("Gray Image", gray_image)
cv2.imshow("Dense SIFT (manual)", image_manual)
cv2.waitKey(0)
cv2.destroyAllWindows()

out_dir = os.path.dirname(__file__)
out_gray = os.path.join(out_dir, "dense_sift_gray.png")
out_kp = os.path.join(out_dir, "dense_sift_keypoints.png")

cv2.imwrite(out_gray, gray_image)
cv2.imwrite(out_kp, image_manual)

print("Saved:", out_gray)
print("Saved:", out_kp)
