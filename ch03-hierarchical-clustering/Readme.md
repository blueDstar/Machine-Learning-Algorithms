# Ch03 — Hierarchical (Agglomerative) Clustering

Thư mục này chứa các ví dụ về **phân cụm phân cấp**:
- Tự cài đặt **Single Linkage**
- Dùng `sklearn.AgglomerativeClustering` (Ward linkage) để phân cụm dữ liệu 2D
- Ứng dụng Agglomerative để **phân đoạn ảnh** và lưu kết quả

## Files
- `custom_single_linkage_clustering.py`: single linkage (tự cài đặt)
- `sklearn_agglomerative_ward_k2.py`: Agglomerative (Ward) với `k=2` cho dữ liệu 2D + plot
- `sklearn_agglomerative_segmentation_k4.py`: phân đoạn ảnh bằng Agglomerative (Ward), resize 128×128, `k=4`, hiển thị và lưu ảnh

## Kết quả (Image Segmentation)
Ảnh phân đoạn (k=4):
![Agglomerative Segmentation k=4](agglomerative_seg_k4_128x128.png)

So sánh ảnh gốc và ảnh phân đoạn:
![Side by side](agglomerative_side_by_side_k4_128x128.png)

## Chạy nhanh
```bash
pip install numpy matplotlib scikit-learn opencv-python scipy
python sklearn_agglomerative_segmentation_k4.py
------------------------------------------------------------------------------
EN — Chapter 03 — Hierarchical (Agglomerative) Clustering

This folder contains hierarchical clustering demos:

Custom Single Linkage

sklearn.AgglomerativeClustering (Ward linkage) on a 2D dataset

Image segmentation with Agglomerative clustering + saved outputs