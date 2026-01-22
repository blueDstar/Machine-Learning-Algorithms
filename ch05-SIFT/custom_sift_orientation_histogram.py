import cv2
import numpy as np

image = cv2.imread(r"F:\Githubrestore\Segmentation-main\Segmentation-main\lena.tif")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

step = 10
size = 20

def compute_gradients(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize =3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize =3)
    return dx, dy

def compute_magnitude_and_orientation(dx, dy):
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360
    return magnitude, orientation

def compute_sift_region(magnitude, orientation, x, y, size):   
    region_magnitude = magnitude[y:y+size, x:x+size]
    region_orientation = orientation[y:y+size, x:x+size]
    bins = 36
    hist = np.zeros(bins)
    angle_unit = 360 / bins
    for i in range(size):
        for j in range(size):
            gradient_angle = region_orientation[i, j]
            gradient_magnitude = region_magnitude[i, j]
            min_angle = int(gradient_angle / angle_unit) % bins
            max_angle = (min_angle + 1) % bins
            mod = gradient_angle % angle_unit
            hist[min_angle] += (gradient_magnitude * (1 - (mod / angle_unit)))
            hist[max_angle] += (gradient_magnitude * (mod / angle_unit))
    return hist



