import gc
import json
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

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
from pprint import pformat
from tqdm import tqdm

sys.path.append(Path(__file__).absolute().parent.parent)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*DataFrameGroupBy\\.apply operated on the grouping columns.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*default of observed=False is deprecated.*"
)

def now():
    now = datetime.now()
    yyyymmdd_hhmmss_part = now.strftime('%Y-%m-%d %H:%M:%S')
    ms_part = f'{int(now.microsecond / 1000):03d}'
    return f'{yyyymmdd_hhmmss_part},{ms_part}'

def smart_categorical_encode(X: cudf.DataFrame, y: cudf.Series) -> tuple:
    # Handle categorical columns in X
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        num_unique = X[col].nunique()
        if num_unique <= 3:
            dummies = cudf.get_dummies(X[col], prefix=col)
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
    tqdm.write(f"[{now()}] Computing ANOVA F complexity (CPU)...")

    # Convert cuDF to NumPy arrays for scikit-learn
    if hasattr(X, 'to_numpy'):  # cuDF DataFrame
        X_np = X.to_numpy()
    else:  # Already pandas DataFrame
        X_np = X.values

    if hasattr(y, 'to_numpy'):  # cuDF Series
        y_np = y.to_numpy()
    else:  # Already pandas Series
        y_np = y.values

    f_scores, _ = f_classif(X_np, y_np)  # Fixed the f*classif typo
    # Replace inf and nan with 0
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return get_stats(cp.asarray(f_scores))

# 1b. Feature relevance (Mutual Information)
def compute_mutual_info_complexity(X: pd.DataFrame, y: pd.Series) -> dict:
    tqdm.write(f"[{now()}] Computing mutual information complexity (CPU)...")

    # Convert cuDF to NumPy arrays for scikit-learn
    if hasattr(X, 'to_numpy'):  # cuDF DataFrame
        X_np = X.to_numpy()
    else:  # Already pandas DataFrame
        X_np = X.values

    if hasattr(y, 'to_numpy'):  # cuDF Series
        y_np = y.to_numpy()
    else:  # Already pandas Series
        y_np = y.values

    mi_scores = mutual_info_classif(X_np, y_np, random_state=42, n_jobs=-1)
    return get_stats(cp.asarray(mi_scores))

# 2a. Local overlap (k-Disagreeing Neighbors)
def compute_kdn_complexity(X: cudf.DataFrame, y: cudf.Series, k: int = None) -> dict:
    tqdm.write(f"[{now()}] Computing kdn complexity (GPU)...")

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
    tqdm.write(f"[{now()}] Computing nearest enemy complexity (GPU)...")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if k is None:
        k = int(cp.sqrt(len(X)))
        k = max(2, k)

    nn = NearestNeighbors(n_neighbors=min(k, len(X)))
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    # Handle cuDF DataFrame/Series - convert to CuPy arrays
    if isinstance(distances, (cudf.DataFrame, cudf.Series)):
        distances_cp = distances.values
    else:
        distances_cp = cp.asarray(distances)

    if isinstance(indices, (cudf.DataFrame, cudf.Series)):
        indices_cp = indices.values
    else:
        indices_cp = cp.asarray(indices)

    y_cp = y.to_cupy()
    enemy_distances = cp.full(len(X), cp.nan)

    for i in range(len(X)):
        for dist, idx in zip(distances_cp[i, 1:], indices_cp[i, 1:]):  # Use _cp versions
            if y_cp[idx] != y_cp[i]:
                enemy_distances[i] = dist
                break

    return get_stats(enemy_distances)

# 3. Margin-based hardness (Approximate Margin)
def compute_margin_complexity(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing margin complexity (GPU)...")

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
    tqdm.write(f"[{now()}] Computing intrinsic dimensionality (GPU)...")

    n_samples, n_features = X.shape

    # Handle edge case where we have fewer samples than features
    max_components = min(n_samples, n_features)

    pca = PCA(n_components=max_components)
    pca.fit(X)

    # Convert explained_variance_ratio_ to CuPy array
    explained_variance_ratio = pca.explained_variance_ratio_
    if hasattr(explained_variance_ratio, 'values'):
        variance_ratio_cp = explained_variance_ratio.values
    else:
        variance_ratio_cp = cp.asarray(explained_variance_ratio)

    # Compute cumulative variance
    cumulative_variance = cp.cumsum(variance_ratio_cp)

    # Find number of components needed for threshold
    components_mask = cumulative_variance >= threshold
    if cp.any(components_mask):
        n_components = int(cp.argmax(components_mask)) + 1
    else:
        n_components = len(cumulative_variance)

    intrinsic_dim_percent = n_components / n_features

    return {
        f"n_components_{threshold * 100:.0f}%": n_components,
        "intrinsic_dimensionality_percent": intrinsic_dim_percent,
        "total_features": n_features,
        "max_explained_variance": float(cumulative_variance[-1])  # Add this for debugging
    }

# 5. Class Imbalance
def compute_class_imbalance(y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing class imbalance (GPU)...")

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
    tqdm.write(f"[{now()}] Computing KNN overlap (GPU)...")

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
    tqdm.write(f"[{now()}] Computing class centroid distance (GPU)...")

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
    tqdm.write(f"[{now()}] Computing clusterability (GPU)...")

    score = silhouette_score(X, y)
    return {"silhouette_score": float(score)}

# Master function to compute all complexity measures
def compute_all_complexity_measures(X_pd: pd.DataFrame, y_pd: pd.Series) -> dict:
    # Enable GPU acceleration for cuDF and cuML
    with using_device_type('gpu'):
        # Convert inputs to cuDF for GPU processing
        X_cu = cudf.DataFrame.from_pandas(X_pd) if isinstance(X_pd, pd.DataFrame) else X_pd
        y_cu = cudf.Series.from_pandas(y_pd) if isinstance(y_pd, pd.Series) else y_pd
        X_cu, y_cu = smart_categorical_encode(X_cu, y_cu)

        def safe_call(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tqdm.write(f"Error in {func.__name__}: {e}")
                return {}

        results = {
            'anova_f': safe_call(compute_anova_f_complexity, X_pd, y_pd),
            'mutual_info': safe_call(compute_mutual_info_complexity, X_pd, y_pd),
            'kdn': safe_call(compute_kdn_complexity, X_cu, y_cu),
            'nearest_enemy': {}, # safe_call(compute_nearest_enemy_complexity, X_cu, y_cu),
            'margin': safe_call(compute_margin_complexity, X_cu, y_cu),
            'intrinsic_dimensionality': safe_call(compute_intrinsic_dimensionality, X_cu),
            'class_imbalance': safe_call(compute_class_imbalance, y_cu),
            'knn_overlap': safe_call(compute_knn_overlap, X_cu, y_cu),
            'centroid_distance': safe_call(compute_class_centroid_distance, X_cu, y_cu),
            'clusterability': safe_call(compute_clusterability, X_cu, y_cu),
        }

        return results

if __name__ == "__main__":

    df_dummy = pd.DataFrame(
        np.random.randn(1000, 25),
        columns=[f"feature_{i}" for i in range(1, 26)]
    )
    df_dummy["label"] = np.random.choice([0, 1], size=1000)
    df_dummy.to_parquet("ready/Binary/Dummy_Binary.parquet", index=False)

    df_dummy = pd.DataFrame(
        np.random.randn(1000, 50),
        columns=[f"feature_{i}" for i in range(1, 51)]
    )
    df_dummy["label"] = np.random.choice([0, 1, 2], size=1000)
    df_dummy.to_parquet("ready/Multiclass/Dummy_Multiclass.parquet", index=False)

    candidate_files = list(Path("datasets").rglob("*"))
    for src_path in tqdm(candidate_files, desc='Candidate', leave=False):
        for kind in tqdm(['Binary', 'Multiclass'], desc='Kind', leave=False):
            if (src_path.is_file() and kind in src_path.name and
                ('generated' in str(src_path.absolute().resolve())) and
                ('.parquet' in src_path.name or '.json' in src_path.name)):
                dst_path = Path(os.path.join('ready', kind, src_path.name))
                if dst_path.is_file() and dst_path.exists():
                    dst_path.unlink()
                tqdm.write(f"[{now()}] Moving {src_path} to {dst_path}...")
                os.makedirs(Path(dst_path).parent, exist_ok=True)
                shutil.move(src_path, dst_path)

    def stratified_sample_fraction(df, strata_col, fraction=0.1):
        return df.groupby(strata_col, group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=42)
        )

    ONE_GB = 1 * 1024 * 1024 * 1024 # 1 GB in bytes

    sample_frac = 0.1
    candidate_files = list(Path("ready").rglob("*.parquet"))
    for src_path in tqdm(candidate_files, desc='Candidate', leave=False):
        for kind in tqdm(['Binary', 'Multiclass'], desc='Kind', leave=False):
            try:
                if src_path.is_file() and kind in src_path.name:
                    abs_path = str(src_path.absolute().resolve())
                    dst_path = Path(abs_path.replace(
                        '.parquet', f'_complexity_frac={int(100*sample_frac)}_pct.json')
                    )
                    if src_path.is_file() and src_path.exists() and src_path.stat().st_size > ONE_GB:
                        tqdm.write(f"Skipping {src_path}; daatset is over 1GB.")
                        continue
                    if dst_path.is_file() and dst_path.exists():
                        # dst_path.unlink()
                        tqdm.write(f"Skipping {dst_path}; complexity JSON already exists.")
                        continue
                    # DF (CPU)
                    df_cpu = pd.read_parquet(src_path)
                    shape_before = df_cpu.shape
                    df_cpu = df_cpu.replace([np.inf, -np.inf], np.nan)
                    df_cpu = df_cpu.dropna(axis='columns', how='all')
                    df_cpu = df_cpu.dropna(axis='index', how='any')
                    df_cpu['label'], label_mapping = df_cpu['label'].factorize()
                    df_cpu = stratified_sample_fraction(df_cpu, 'label', fraction=sample_frac)
                    label_mapping_dict = {str(label_mapping[i]): int(i) for i in range(len(label_mapping))}
                    shape_after = df_cpu.shape
                    # DF (GPU)
                    df_gpu = cudf.from_pandas(df_cpu)
                    del df_cpu
                    gc.collect()
                    tqdm.write(f'[{now()}] DS: {str(src_path):<80}' + 
                               f' | SHAPE_0: {str(shape_before):<16}' +
                               f' | SHAPE_1: {str(shape_after):<16}')
                    X, y = df_gpu.drop(columns=['label']), df_gpu['label']
                    X, y = smart_categorical_encode(X, y)
                    # Metrics
                    metrics = compute_all_complexity_measures(X, y)
                    metrics['mappings'] = label_mapping_dict
                    with open(dst_path, 'w') as fp:
                        json.dump(metrics, fp, indent=4)
                    metrics_str = pformat(metrics, compact=True, indent=4)
                    tqdm.write(f'[{now()}] MET: {metrics_str}')
            except Exception as e:
                tqdm.write(f"[{now()}] Error in {src_path}: {e}")
