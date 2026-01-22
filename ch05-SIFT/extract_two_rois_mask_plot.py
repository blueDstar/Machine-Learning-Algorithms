import os
import numpy as np
import matplotlib.pyplot as plt

img_path = r"F:\Githubrestore\Segmentation-main\Segmentation-main\lena.tif"
anh = plt.imread(img_path)

x1, y1 = 715, 445
x2, y2 = 1150, 850

x3, y3 = 830, 55
x4, y4 = 1050, 265

roi1 = anh[y1:y2, x1:x2, :]
roi2 = anh[y3:y4, x3:x4, :]

nenden = np.zeros_like(anh)
nenden[y1:y2, x1:x2] = roi1
nenden[y3:y4, x3:x4] = roi2

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(nenden)
plt.title("ROI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(anh)
plt.title("Gốc")
plt.axis("off")

out_path = os.path.join(os.path.dirname(__file__), "roi_two_regions_result.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
