import sys
import warnings
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.common.device_selection import using_device_type
from cuml.decomposition import PCA
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
from cuml.svm import SVC
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC, LinearSVC

from modules.logging.logger import log_print

sys.path.append(Path(__file__).absolute().parent.parent)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def smart_categorical_encode(X: cudf.DataFrame, y: cudf.Series) -> tuple:
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        num_unique = X[col].nunique()
        if num_unique <= 3:
            dummies = X[col].str.get_dummies(prefix=col, drop_first=True)
            X = X.drop(columns=[col])
            X = X.join(dummies)
        else:
            X[col], _ = X[col].factorize(sort=True)

    return X, y

# Helper: get basic statistics
def get_stats(values: np.ndarray) -> dict:
    return {
        "size": len(values),
        "max": np.max(values),
        "min": np.min(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "Q1": np.percentile(values, 25),
        "Q3": np.percentile(values, 75),
        "Q3-Q1": np.percentile(values, 75) - np.percentile(values, 25),
        "stdev": np.std(values),
        "variance": np.var(values)
    }

# 1a. Feature relevance (ANOVA F-statistic)
def compute_anova_f_complexity(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing ANOVA F complexity...")
    f_scores, _ = f_classif(X, y)

    # Replace inf and nan with 0
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return get_stats(f_scores)

# 1b. Feature relevance (Mutual Information)
def compute_mutual_info_complexity(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing mutual information complexity...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=-1)
    return get_stats(mi_scores)

def compute_kdn_complexity(X: cudf.DataFrame, y: cudf.Series, k: int = None) -> dict:
    log_print("Computing kdn complexity (GPU)...")

    if k is None:
        k = int(cp.sqrt(len(X)))
        k = max(2, k)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)

    neighbors = nn.kneighbors(X, return_indices=True)[1][:, 1:]  # Exclude self
    y_cp = y.to_cupy()

    kdn_scores = cp.mean(y_cp[neighbors] != y_cp[:, None], axis=1)

    return get_stats(cp.asnumpy(kdn_scores))

# 2b. Boundary Density (Nearest Enemy Distance)
def compute_nearest_enemy_complexity(X: cudf.DataFrame, y: cudf.Series, k: int = None) -> dict:
    log_print("Computing nearest enemy complexity (GPU)...")

    if k is None:
        k = int(cp.sqrt(len(X)))
        k = max(2, k)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)

    distances, indices = nn.kneighbors(X, return_distance=True)
    y_cp = y.to_cupy()

    enemy_distances = cp.full(len(X), cp.nan)

    for i in range(len(X)):
        for dist, idx in zip(distances[i, 1:], indices[i, 1:]):  # Skip self
            if y_cp[idx] != y_cp[i]:
                enemy_distances[i] = dist
                break

    return get_stats(cp.asnumpy(enemy_distances))

# 3. Margin-based hardness (Approximate Margin)
def compute_margin_complexity(X: cudf.DataFrame, y: cudf.Series) -> dict:
    log_print("Computing margin complexity (GPU)...")

    clf = SVC(kernel='linear', probability=False, max_iter=20000)
    clf.fit(X, y)

    # Decision function returns signed distance to hyperplane
    decision_function = clf.decision_function(X)

    # Margin = abs(distance to decision boundary)
    margins = cp.abs(decision_function)

    return get_stats(cp.asnumpy(margins))

# 4. Intrinsic Dimensionality (PCA variance ratio)
def compute_intrinsic_dimensionality(X: cudf.DataFrame, threshold: float = 0.95) -> dict:
    log_print("Computing intrinsic dimensionality (GPU)...")

    pca = PCA(random_state=42)
    pca.fit(X)

    cumulative_variance = cp.cumsum(pca.explained_variance_ratio_)
    n_components = int(cp.searchsorted(cumulative_variance, threshold)) + 1
    n_features = X.shape[1]

    intrinsic_dim_percent = n_components / n_features

    return {
        f"n_components_{threshold * 100}%": n_components,
        "intrinsic_dimensionality_percent": intrinsic_dim_percent
    }

# 5. Class Imbalance
def compute_class_imbalance(y: cudf.Series) -> dict:
    log_print("Computing class imbalance (GPU)...")

    class_counts = y.value_counts(sort=False)
    n_classes = len(class_counts)
    max_count = int(class_counts.max())
    min_count = int(class_counts.min())

    imbalance_ratio = max_count / min_count

    probs = class_counts.values / class_counts.values.sum()
    entropy = -cp.sum(probs * cp.log(probs))
    normalized_entropy = float(entropy / cp.log(n_classes)) if n_classes > 1 else 1.0

    return {
        "n_classes": n_classes,
        "imbalance_ratio": imbalance_ratio,
        "normalized_entropy": normalized_entropy
    }

# 6. KNN Overlap Fraction
def compute_knn_overlap(X: cudf.DataFrame, y: cudf.Series, k: int = None) -> dict:
    log_print("Computing KNN overlap (GPU)...")

    if k is None:
        k = int(cp.sqrt(len(X)))
        k = max(2, k)

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    # Ensure both y and y_pred are on GPU and compatible
    y_cp = y.to_cupy()
    y_pred_cp = y_pred.to_cupy()

    accuracy = cp.mean(y_pred_cp == y_cp)
    overlap_fraction = float(1 - accuracy)

    return {"knn_overlap_fraction": overlap_fraction}

# 7. Class Centroid Distance
def compute_class_centroid_distance(X: cudf.DataFrame, y: cudf.Series) -> dict:
    log_print("Computing class centroid distance (GPU)...")

    # Compute class centroids
    X_with_labels = X.copy()
    X_with_labels['label'] = y
    class_centroids = X_with_labels.groupby('label').mean().drop(columns=['label'])

    # Convert to cupy arrays
    centroids_cp = class_centroids.to_cupy()

    # Compute pairwise squared distances
    diff = centroids_cp[:, None, :] - centroids_cp[None, :, :]
    dists = cp.sqrt(cp.sum(diff ** 2, axis=-1))

    # Extract upper triangle without diagonal
    upper_triangle = dists[cp.triu_indices(len(centroids_cp), k=1)]
    mean_distance = float(cp.mean(upper_triangle))

    return {"class_centroid_mean_distance": mean_distance}

# 8. Global Clusterability (Silhouette Score)
def compute_clusterability(X: cudf.DataFrame, y: cudf.Series) -> dict:
    log_print("Computing clusterability (GPU)...")

    score = silhouette_score(X, y)
    return {"silhouette_score": float(score)}

# Master function to compute all complexity measures
def compute_all_complexity_measures(X_pd: pd.DataFrame, y_pd: pd.Series) -> dict:
    # Enable GPU acceleration for cuDF and cuML
    with using_device_type('gpu'):
        # Convert inputs to cuDF for GPU processing
        X_cu = cudf.DataFrame.from_pandas(X_pd)
        y_cu = cudf.Series(y_pd)

        def safe_call(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_print(f"Error in {func.__name__}: {e}")
                return float('-inf')

        return {
            'anova_f': safe_call(compute_anova_f_complexity, X_pd, y_pd),
            'mutual_info': safe_call(compute_mutual_info_complexity, X_pd, y_pd),
            'kdn': safe_call(compute_kdn_complexity, X_cu, y_cu),
            'nearest_enemy': safe_call(compute_nearest_enemy_complexity, X_cu, y_cu),
            'margin': safe_call(compute_margin_complexity, X_cu, y_cu),
            'intrinsic_dimensionality': safe_call(compute_intrinsic_dimensionality, X_cu),
            'class_imbalance': safe_call(compute_class_imbalance, y_cu),
            'knn_overlap': safe_call(compute_knn_overlap, X_cu, y_cu),
            'centroid_distance': safe_call(compute_class_centroid_distance, X_cu, y_cu),
            'clusterability': safe_call(compute_clusterability, X_cu, y_cu),
        }
