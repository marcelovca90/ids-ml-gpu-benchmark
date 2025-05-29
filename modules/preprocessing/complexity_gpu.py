import sys
import warnings
from pathlib import Path
from pprint import pprint

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.common.device_selection import using_device_type
from cuml.decomposition import PCA
from cuml.metrics.cluster import silhouette_score
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
from cuml.svm import LinearSVC
from sklearn.feature_selection import f_classif, mutual_info_classif

from modules.logging.logger import log_print

sys.path.append(Path(__file__).absolute().parent.parent)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def smart_categorical_encode(X: cudf.DataFrame, y: cudf.Series) -> tuple:
    # Handle categorical columns in X
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        num_unique = X[col].nunique()
        if num_unique <= 3:
            dummies = X[col].str.get_dummies()
            dummies = dummies.rename(columns=lambda c: f"{col}_{c}")
            X = X.drop(columns=[col])
            X = X.join(dummies)
        else:
            X[col], _ = X[col].factorize(sort=True)

    # Handle categorical/string labels in y
    if y.dtype == 'object' or str(y.dtype) == 'category':
        # For categorical or string data, convert to numeric codes
        if hasattr(y, 'cat'):
            # Already categorical
            y_encoded = y.cat.codes
        else:
            # Convert strings to categorical first, then get codes
            y_categorical = y.astype('category')
            y_encoded = y_categorical.cat.codes

        # Return the encoded y series
        return X, y_encoded

    # If y is already numeric, return as-is
    return X, y

# Helper: get basic statistics
def get_stats(values: cp.ndarray) -> dict:
    return {
        "size": int(values.size),
        "max": float(cp.max(values)),
        "min": float(cp.min(values)),
        "mean": float(cp.mean(values)),
        "median": float(cp.median(values)),
        "Q1": float(cp.percentile(values, 25)),
        "Q3": float(cp.percentile(values, 75)),
        "Q3-Q1": float(cp.percentile(values, 75) - cp.percentile(values, 25)),
        "stdev": float(cp.std(values)),
        "variance": float(cp.var(values))
    }

# 1a. Feature relevance (ANOVA F-statistic)
def compute_anova_f_complexity(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing ANOVA F complexity (CPU)...")
    f_scores, _ = f_classif(X, y)

    # Replace inf and nan with 0
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return get_stats(cp.asarray(f_scores))

# 1b. Feature relevance (Mutual Information)
def compute_mutual_info_complexity(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing mutual information complexity (CPU)...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=-1)
    return get_stats(cp.asarray(mi_scores))

# 2a. Local overlap (k-Disagreeing Neighbors)
def compute_kdn_complexity(X: cudf.DataFrame, y: cudf.Series, k: int = None) -> dict:
    log_print("Computing kdn complexity (GPU)...")

    if k is None:
        k = int(cp.sqrt(len(X)))
        k = max(2, k)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)

    # Get distances and indices; exclude self-neighbor (first column)
    distances, indices = nn.kneighbors(X, return_distance=True)

    # Handle cuDF DataFrame - convert to CuPy array first
    if isinstance(indices, cudf.DataFrame):
        indices_cp = indices.values  # Get underlying CuPy array
    else:
        indices_cp = cp.asarray(indices)

    neighbors = indices_cp[:, 1:]

    # Reset index to ensure proper alignment
    y_reset = y.reset_index(drop=True)
    y_cp = y_reset.to_cupy()

    neighbor_labels = y_cp[neighbors]  # shape: (n_samples, k)
    instance_labels = y_cp[:, None]    # shape: (n_samples, 1)

    disagreement = neighbor_labels != instance_labels
    kdn_scores = cp.mean(disagreement, axis=1)

    return get_stats(kdn_scores)

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

    return get_stats(enemy_distances)

# 3. Margin-based hardness (Approximate Margin)
def compute_margin_complexity(X: cudf.DataFrame, y: cudf.Series) -> dict:
    log_print("Computing margin complexity (GPU)...")

    # Use cuML's LinearSVC for GPU acceleration
    clf = LinearSVC(
        C=1.0,                    # Regularization parameter
        max_iter=20000,           # Maximum iterations
        tol=1e-4,                 # Tolerance for stopping criterion
        fit_intercept=True,       # Whether to fit intercept
        verbose=False             # Suppress output
    )

    clf.fit(X, y)

    # Decision function returns signed distance to hyperplane
    decision_function = clf.decision_function(X)

    # Handle multi-class case (decision_function returns multiple columns)
    if hasattr(decision_function, 'shape') and len(decision_function.shape) > 1 and decision_function.shape[1] > 1:
        # For multi-class, take minimum absolute distance across all decision boundaries
        if hasattr(decision_function, 'values'):
            decision_values = decision_function.values
        else:
            decision_values = cp.asarray(decision_function)
        margins = cp.min(cp.abs(decision_values), axis=1)
    else:
        # Binary classification case
        if hasattr(decision_function, 'values'):
            decision_values = decision_function.values
        else:
            decision_values = cp.asarray(decision_function)
        margins = cp.abs(decision_values)

    return get_stats(margins)

# 4. Intrinsic Dimensionality (PCA variance ratio)
def compute_intrinsic_dimensionality(X: cudf.DataFrame, threshold: float = 0.95) -> dict:
    log_print("Computing intrinsic dimensionality (GPU)...")

    pca = PCA(n_components=int(X.shape[1]))
    pca.fit(X)

    # Convert explained_variance_ratio_ to CuPy array
    explained_variance_ratio = pca.explained_variance_ratio_
    if hasattr(explained_variance_ratio, 'values'):
        # It's a cuDF Series, get the underlying CuPy array
        variance_ratio_cp = explained_variance_ratio.values
    else:
        # Convert to CuPy array
        variance_ratio_cp = cp.asarray(explained_variance_ratio)

    # Compute cumulative variance
    cumulative_variance = cp.cumsum(variance_ratio_cp)

    # Find number of components needed for threshold (alternative approach)
    # Find first index where cumulative variance >= threshold
    components_mask = cumulative_variance >= threshold
    if cp.any(components_mask):
        n_components = int(cp.argmax(components_mask)) + 1
    else:
        # If threshold is never reached, use all components
        n_components = len(cumulative_variance)
    n_features = X.shape[1]
    intrinsic_dim_percent = n_components / n_features

    return {
        f"n_components_{threshold * 100:.0f}%": n_components,
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
# 7. Class Centroid Distance
def compute_class_centroid_distance(X: cudf.DataFrame, y: cudf.Series) -> dict:
    log_print("Computing class centroid distance (GPU)...")

    # Compute class centroids
    X_with_labels = X.copy()
    X_with_labels['label'] = y

    # Group by label and compute mean (label becomes index, not column)
    class_centroids = X_with_labels.groupby('label').mean()

    # Convert to cupy arrays
    if hasattr(class_centroids, 'values'):
        centroids_cp = class_centroids.values
    else:
        centroids_cp = class_centroids.to_cupy()

    # Ensure it's a CuPy array
    centroids_cp = cp.asarray(centroids_cp)

    # Handle edge case: only one class
    if len(centroids_cp) < 2:
        return {"class_centroid_mean_distance": 0.0}

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
        y_cu = cudf.Series.from_pandas(y_pd)
        X_cu, y_cu = smart_categorical_encode(X_cu, y_cu)

        def safe_call(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_print(f"Error in {func.__name__}: {e}")
                return {}

        results = {
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

        pprint(results, indent=4)

        return results
