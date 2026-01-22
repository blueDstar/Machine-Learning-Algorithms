# ML & CV Algorithms — From Scratch + Sklearn/OpenCV

Repo này tổng hợp các bài thực hành (dạng “chapter”) về **Machine Learning & Computer Vision**:  
K-Means, Hierarchical Clustering, KNN, SIFT, Naive Bayes (và các demo liên quan).

Mỗi chương là một thư mục riêng, có `README.md` bên trong để mô tả chi tiết + ảnh minh hoạ.

---

## Cấu trúc repo

- `ch01-Kmeans/` — K-Means (tự cài đặt + sklearn) + phân đoạn ảnh RGB
- `ch02-hierarchical-clustering/` — Hierarchical Clustering (dendrogram, linkage, fcluster…)
- `ch03-hierarchical-clustering/` — Hierarchical Clustering (ứng dụng/biến thể khác, gồm phân đoạn ảnh bằng Agglomerative)
- `ch04-KNN/` — KNN (custom + sklearn, so sánh K, CV error rate, plots)
- `ch05-SIFT/` — SIFT & thao tác ảnh (Dense SIFT manual, OpenCV SIFT keypoints, ROI, downsample…)
- `ch06-Naive-Bayes/` — Naive Bayes cho hand gesture + HOG (grid search, learning curve, lưu model…)
- `lena.tif` — ảnh mẫu dùng cho một số bài (ví dụ segmentation)

> Gợi ý: mỗi chương nên có `img/` hoặc `outputs/` để chứa ảnh kết quả, giúp repo gọn hơn.

---

## Yêu cầu môi trường

```bash
pip install numpy matplotlib scikit-learn opencv-python seaborn pandas scikit-image joblib scipy
--------------------------------------------------------------------------------
EN — ML & CV Algorithms — From Scratch + Sklearn/OpenCV

This repository is a collection of “chapter-style” labs for Machine Learning & Computer Vision:
K-Means, Hierarchical Clustering, KNN, SIFT, Naive Bayes (plus related demos).

Each chapter has its own folder and an internal README.md describing scripts and embedded outputs.

Repository structure

ch01-Kmeans/ — K-Means (from scratch + sklearn) + RGB image segmentation

ch02-hierarchical-clustering/ — Hierarchical clustering (dendrogram, linkage comparison, fcluster…)

ch03-hierarchical-clustering/ — Additional hierarchical clustering demos (incl. image segmentation with agglomerative)

ch04-KNN/ — KNN (custom + sklearn, K comparison, CV error rate, plots)

ch05-SIFT/ — SIFT & image operations (manual dense SIFT, OpenCV SIFT keypoints, ROI, downsample…)

ch06-Naive-Bayes/ — Naive Bayes for hand gestures + HOG (grid search, learning curve, model saving…)

lena.tif — sample image used in multiple demos
Requirements:
pip install numpy matplotlib scikit-learn opencv-python seaborn pandas scikit-image joblib scipy
