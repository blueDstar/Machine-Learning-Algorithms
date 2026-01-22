# Ch01 — K-Means

Thư mục này gồm các ví dụ **K-Means** (tự cài đặt và dùng `scikit-learn`) trên dữ liệu 1D/2D và ứng dụng phân đoạn ảnh.

## Cấu trúc
- `img/lena.tif`: ảnh mẫu cho bài phân đoạn ảnh
- `K_means_1d_demo.py`: K-Means, demo cơ bản
- `kmeans_1d_convergence_demo.py`: K-Means, in chi tiết các lần lặp và kiểm tra hội tụ
- `kmeans_2d_demo.py`: K-Means  demo dữ liệu 2 chiều
- `sklearn_kmeans_1d_manual_init.py`: K-Means 1D (sklearn), khởi tạo tâm cụm thủ công (`init`)
- `sklearn_kmeans_2d_manual_init.py`: K-Means 2D (sklearn), khởi tạo tâm cụm thủ công (`init`)
- `kmeans_rgb_image_segmentation.py`: Phân đoạn ảnh bằng K-Means trên không gian màu RGB (dùng `img/lena.tif`)
- `kmeans_rgb_image_segmentation_k2.py`: Phân đoạn ảnh RGB bằng K-Means với **k = 2**
- `kmeans_rgb_image_segmentation_k3.py`: Phân đoạn ảnh RGB bằng K-Means với **k = 3**

## Chạy nhanh
```bash
pip install numpy matplotlib scikit-learn opencv-python
python K_means_1d_demo.py
python kmeans_rgb_image_segmentation.py

--------------------------------------------------------------------------------------

This folder contains K-Means examples (from-scratch + scikit-learn) for 1D/2D data and an RGB image segmentation demos.

img/lena.tif: sample image for segmentation


K_means_1d_demo.py: 1D K-Means (from scratch) basic demo

kmeans_1d_convergence_demo.py: 1D K-Means (from scratch) with iteration logs + convergence check

kmeans_2d_demo.py: 2D K-Means (from scratch) demo

sklearn_kmeans_1d_manual_init.py: 1D K-Means (scikit-learn) with manual centroid initialization

sklearn_kmeans_2d_manual_init.py: 2D K-Means (scikit-learn) with manual centroid initialization

kmeans_rgb_image_segmentation.py: RGB image segmentation using K-Means (reads img/lena.tif)

kmeans_rgb_image_segmentation_k2.py: RGB image segmentation using K-Means with k = 2

kmeans_rgb_image_segmentation_k3.py: RGB image segmentation using K-Means with k = 3