# Ch02 — Hierarchical Clustering

Thư mục này chứa các ví dụ về **phân cụm phân cấp (Hierarchical / Agglomerative Clustering)**, bao gồm: vẽ dendrogram, so sánh các kiểu linkage và cắt cây để lấy cụm.

## Cấu trúc & file

- `plot_2d_points_labeled_grid.py`  
  Vẽ dữ liệu 2D lên lưới và gắn nhãn điểm (p1, p2, ...).  
  Dùng để quan sát trực quan trước khi phân cụm.

- `dendrogram_linkage_comparison.py`  
  So sánh **nhiều phương pháp liên kết (linkage)** bằng cách vẽ dendrogram cho từng method:  
  `single`, `complete`, `centroid`, `average`, `ward`.

- `single_linkage_fcluster.py`  
  Phân cụm phân cấp với **single linkage**, sau đó dùng `fcluster` để **cắt cây** và lấy nhãn cụm theo số cụm mong muốn (`criterion='maxclust'`).  
  Có kèm dendrogram minh hoạ.

- `agglomerative_ward_dendrogram_and_clusters.py`  
  Vẽ dendrogram với **Ward linkage** (SciPy) và phân cụm bằng `sklearn.AgglomerativeClustering` (Ward + Euclidean).  
  Cuối cùng vẽ scatter plot theo nhãn cụm.

## Chạy nhanh
```bash
pip install numpy matplotlib scipy scikit-learn
python plot_2d_points_labeled_grid.py
python dendrogram_linkage_comparison.py
python single_linkage_fcluster.py
python agglomerative_ward_dendrogram_and_clusters.py

----------------------------------------------------------------------------------
EN — Chapter 02 — Hierarchical Clustering

This folder contains Hierarchical/Agglomerative Clustering demos: dendrogram visualization, linkage method comparison, and cutting the tree to obtain cluster labels.

Files

plot_2d_points_labeled_grid.py: plot and label the 2D dataset on a grid.

dendrogram_linkage_comparison.py: compare dendrograms across linkage methods (single, complete, centroid, average, ward).

single_linkage_fcluster.py: single-linkage hierarchy + fcluster tree cut (criterion='maxclust') + dendrogram.

agglomerative_ward_dendrogram_and_clusters.py: Ward dendrogram (SciPy) + clustering with sklearn.AgglomerativeClustering, then scatter plot by cluster labels.
