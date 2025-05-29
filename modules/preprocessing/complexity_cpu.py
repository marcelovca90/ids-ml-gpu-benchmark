import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

# Helper: apply categorical encoding (label or one-hot based on no. uniques)
def smart_categorical_encode(X: pd.DataFrame, y: pd.Series) -> dict:
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        num_unique = X[col].nunique()
        if num_unique <= 3:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = X.drop(columns=[col])
            X = pd.concat([X, dummies], axis=1)
        else:
            X[col], _ = pd.factorize(X[col], sort=True)
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

# 2a. Local overlap (k-Disagreeing Neighbors)
def compute_kdn_complexity(X: pd.DataFrame, y: pd.Series, k: int = None) -> dict:
    log_print("Computing kdn complexity...")
    if k is None:
        k = int(np.sqrt(len(X)))
        k = max(2, k)  # Ensure at least 2

    nn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X)  # << parallelized
    neighbors = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self

    kdn_scores = []
    for idx, neighbor_indices in enumerate(neighbors):
        neighbor_labels = y.iloc[neighbor_indices]
        disagreement = np.mean(neighbor_labels != y.iloc[idx])
        kdn_scores.append(disagreement)

    return get_stats(np.array(kdn_scores))

# 2b. Boundary Density (Nearest Enemy Distance)
def compute_nearest_enemy_complexity(X: pd.DataFrame, y: pd.Series, k: int = None) -> dict:
    log_print("Computing nearest enemy complexity...")
    if k is None:
        k = int(np.sqrt(len(X)))
        k = max(2, k)

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)  # << parallelized

    enemy_distances = []

    for i in range(len(X)):
        distances, indices = nn.kneighbors(X.iloc[[i]], n_neighbors=k)
        neighbors_y = y.iloc[indices[0]]
        neighbors_distances = distances[0]

        for dist, label in zip(neighbors_distances[1:], neighbors_y[1:]):  # skip self
            if label != y.iloc[i]:
                enemy_distances.append(dist)
                break

    return get_stats(np.array(enemy_distances))

# 3. Margin-based hardness (Approximate Margin)
def compute_margin_complexity(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing margin complexity...")
    # Use a linear SVM to approximate margins
    clf = SVC(kernel='linear', probability=False, 
              cache_size=1024, random_state=42)
    clf.fit(X, y)
    decision_function = clf.decision_function(X)

    if len(decision_function.shape) == 1:  # binary case
        margins = np.abs(decision_function)
    else:  # multiclass case
        margins = np.min(decision_function, axis=1)

    return get_stats(margins)

def compute_margin_complexity_fast(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing margin complexity (LinearSVC version)...")
    clf = LinearSVC(dual=False, random_state=42, max_iter=20000)
    clf.fit(X, y)

    # Calculate margin = distance to the decision boundary
    decision_function = clf.decision_function(X)

    if len(decision_function.shape) == 1:  # binary
        margins = np.abs(decision_function)
    else:  # multiclass (one-vs-rest margins)
        margins = np.min(decision_function, axis=1)

    return get_stats(margins)

# 4. Intrinsic Dimensionality (PCA variance ratio)
def compute_intrinsic_dimensionality(X: pd.DataFrame, threshold: float = 0.95) -> dict:
    log_print("Computing intrinsic dimensionality...")
    pca = PCA(random_state=42)
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    n_components = np.searchsorted(cumulative_variance, threshold) + 1
    n_features = X.shape[1]

    intrinsic_dim_percent = n_components / n_features

    return {
        "n_components_95%": n_components,
        "intrinsic_dimensionality_percent": intrinsic_dim_percent  # << NEW
    }

# 5. Class Imbalance
def compute_class_imbalance(y: pd.Series) -> dict:
    log_print("Computing class imbalance...")
    class_counts = y.value_counts()
    n_classes = len(class_counts)
    max_count = class_counts.max()
    min_count = class_counts.min()

    imbalance_ratio = max_count / min_count

    # Normalized entropy
    probs = class_counts / class_counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    normalized_entropy = entropy / np.log(n_classes) if n_classes > 1 else 1.0

    return {
        "n_classes": n_classes,
        "imbalance_ratio": imbalance_ratio,
        "normalized_entropy": normalized_entropy
    }

# 6. KNN Overlap Fraction
def compute_knn_overlap(X: pd.DataFrame, y: pd.Series, k: int = None) -> dict:
    log_print("Computing KNN overlap...")
    if k is None:
        k = int(np.sqrt(len(X)))
        k = max(2, k)

    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    accuracy = np.mean(y_pred == y)
    overlap_fraction = 1 - accuracy

    return {"knn_overlap_fraction": overlap_fraction}

# 7. Class Centroid Distance
def compute_class_centroid_distance(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing class centroid distance...")
    class_centroids = X.groupby(y).mean()
    dists = cdist(class_centroids, class_centroids)
    upper_triangle = dists[np.triu_indices_from(dists, k=1)]  # Take distances above diagonal
    mean_distance = np.mean(upper_triangle)
    return {"class_centroid_mean_distance": mean_distance}

# 8. Global Clusterability (Silhouette Score)
def compute_clusterability(X: pd.DataFrame, y: pd.Series) -> dict:
    log_print("Computing clusterability (silhouette score)...")
    try:
        # Use true labels for silhouette (clusters = classes)
        score = silhouette_score(X, y, random_state=42)
    except Exception:
        # If silhouette computation fails (e.g., only one class), set to NaN
        score = np.nan
    return {"silhouette_score": score}

# Master function to compute all complexity measures
def compute_all_complexity_measures(X: pd.DataFrame, y: pd.Series) -> dict:
    def safe_call(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_print(f"Error in {func.__name__}: {e}")
            return float('-inf')  # Use -inf to indicate failure
    return {
        'anova_f': safe_call(compute_anova_f_complexity, X, y),
        'mutual_info': safe_call(compute_mutual_info_complexity, X, y),
        'kdn': safe_call(compute_kdn_complexity, X, y),
        'nearest_enemy': safe_call(compute_nearest_enemy_complexity, X, y),
        'margin': safe_call(compute_margin_complexity_fast, X, y),
        'intrinsic_dimensionality': safe_call(compute_intrinsic_dimensionality, X),
        'class_imbalance': safe_call(compute_class_imbalance, y),
        'knn_overlap': safe_call(compute_knn_overlap, X, y),
        'centroid_distance': safe_call(compute_class_centroid_distance, X, y),
        'clusterability': safe_call(compute_clusterability, X, y),
    }
