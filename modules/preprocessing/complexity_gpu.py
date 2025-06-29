import gc
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.decomposition import PCA
from cuml.metrics.cluster import silhouette_score
from cuml.neighbors import NearestNeighbors
from cuml.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
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

def stratified_sample_with_min(df: cudf.DataFrame, stratify_col: str, max_total_samples: int, min_samples_per_class: int) -> cudf.DataFrame:
    # Step 0: Oversample minority classes if needed
    class_counts = df[stratify_col].value_counts()
    minority_classes = class_counts[class_counts < min_samples_per_class]

    if len(minority_classes) > 0:
        oversampled = []
        for cls in minority_classes.index.to_pandas():
            samples = df[df[stratify_col] == cls].sample(n=min_samples_per_class, replace=True)
            oversampled.append(samples)
        df_balanced = cudf.concat([df] + oversampled, ignore_index=True)
    else:
        df_balanced = df

    # If total size is already under the threshold, return full balanced set
    if len(df_balanced) <= max_total_samples:
        return df_balanced

    # Step 1: Recompute proportions
    class_proportions = df_balanced[stratify_col].value_counts(normalize=True)

    # Step 2: Initial per-class sample counts
    sample_counts = (class_proportions * max_total_samples).round().astype('int32')

    # Step 3: Enforce min_samples_per_class
    min_val = sample_counts.dtype.type(min_samples_per_class)
    sample_counts[sample_counts < min_val] = min_val

    # Step 4: Re-normalize if total > max_total_samples
    total = sample_counts.sum()
    if total > max_total_samples:
        scale = max_total_samples / total
        sample_counts = (sample_counts * scale).round().astype('int32')
        min_val = sample_counts.dtype.type(min_samples_per_class)
        sample_counts[sample_counts < min_val] = min_val

    # Step 5: Final stratified sampling
    sampled = []
    for cls, count in zip(sample_counts.index.to_pandas(), sample_counts.values_host):
        group = df_balanced[df_balanced[stratify_col] == cls]
        count = min(count, len(group))
        sampled.append(group.sample(n=count, replace=False))

    return cudf.concat(sampled, ignore_index=True)

# 1a. Feature relevance (ANOVA F-statistic)
def compute_anova_f_complexity(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing ANOVA F complexity (GPU)...")

    unique_labels = y.unique()
    n_classes = len(unique_labels)
    n_samples = len(X)
    n_features = X.shape[1]

    # Overall mean per feature
    overall_mean = X.mean()

    # Between-group and within-group variance
    ss_between = cp.zeros(n_features)
    ss_within = cp.zeros(n_features)

    for cls in unique_labels.values_host:
        cls_mask = (y == cls)
        X_cls = X[cls_mask]
        n_cls = len(X_cls)

        if n_cls < 2:
            continue  # Avoid divide-by-zero

        cls_mean = X_cls.mean()
        ss_between += n_cls * ((cls_mean - overall_mean) ** 2).to_cupy()
        ss_within += ((X_cls - cls_mean) ** 2).sum().to_cupy()

    # Degrees of freedom
    df_between = n_classes - 1
    df_within = n_samples - n_classes

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_scores = cp.divide(ms_between, ms_within)
    f_scores = cp.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return get_stats(f_scores)

# 1b. Feature relevance (Mutual Information)
def compute_mutual_info_complexity(X: cudf.DataFrame, y: cudf.Series, max_bins: int = 64) -> dict:
    tqdm.write(f"[{now()}] Computing mutual information complexity (GPU, approx)...")

    X_cp = X.to_cupy()
    y_cp = y.to_cupy()
    n_samples, n_features = X_cp.shape

    n_bins = int(cp.clip(cp.sqrt(n_samples), 10, max_bins))
    mi_scores = cp.zeros(n_features)

    # Precompute y histogram
    unique_y = cp.unique(y_cp)
    n_y_bins = unique_y.size
    y_hist, _ = cp.histogram(y_cp, bins=n_y_bins)
    py = y_hist / cp.sum(y_hist)

    for i in range(n_features):
        xi = X_cp[:, i]

        # Discretize feature
        xi_hist, bin_edges = cp.histogram(xi, bins=n_bins)
        px = xi_hist / cp.sum(xi_hist)

        # Joint histogram
        joint_hist, _, _ = cp.histogram2d(xi, y_cp, bins=(bin_edges, n_y_bins))
        pxy = joint_hist / cp.sum(joint_hist)

        # Avoid invalid log values explicitly
        denominator = px[:, None] * py[None, :]
        valid = (pxy > 0) & (denominator > 0)
        log_term = cp.zeros_like(pxy)
        log_term[valid] = cp.log(pxy[valid] / denominator[valid])

        mi = cp.sum(pxy * log_term)
        mi_scores[i] = mi

    return get_stats(mi_scores)

# 2a. Local overlap (k-Disagreeing Neighbors)
# 2b. Boundary Density (Nearest Enemy Distance)
# 6. KNN Overlap Fraction
def compute_local_overlap_and_knn_metrics(X: cudf.DataFrame, y: cudf.Series, k: int = None) -> dict:
    tqdm.write(f"[{now()}] Computing KDN, nearest enemy, and KNN overlap (GPU)...")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if k is None:
        n_classes = y.nunique()
        k = min(50, max(2 * n_classes, 10))

    # Fit neighbors once
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)))
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    distances_cp = distances.values if isinstance(distances, (cudf.DataFrame, cudf.Series)) else cp.asarray(distances)
    indices_cp = indices.values if isinstance(indices, (cudf.DataFrame, cudf.Series)) else cp.asarray(indices)

    neighbors = indices_cp[:, 1:]
    neighbor_dists = distances_cp[:, 1:]

    y_cp = y.to_cupy()
    instance_labels = y_cp[:, None]
    neighbor_labels = y_cp[neighbors]

    # KDN
    disagreements = neighbor_labels != instance_labels
    kdn_scores = cp.mean(disagreements, axis=1)

    # Nearest Enemy Distance
    enemy_distances = cp.full(len(X), cp.nan)
    for i in range(len(X)):
        for dist, lbl in zip(neighbor_dists[i], neighbor_labels[i]):
            if lbl != y_cp[i]:
                enemy_distances[i] = dist
                break

    # KNN Overlap: majority vote from k neighbors
    # Predict majority label per row
    def row_mode(arr):
        # Vectorized mode using bincount; works with int labels
        counts = cp.apply_along_axis(lambda x: cp.bincount(x, minlength=y_cp.max()+1), axis=1, arr=arr)
        return cp.argmax(counts, axis=1)

    y_pred_knn = row_mode(neighbor_labels)
    knn_overlap_fraction = float(1 - cp.mean(y_pred_knn == y_cp))

    return {
        "kdn": get_stats(kdn_scores),
        "nearest_enemy": get_stats(enemy_distances),
        "knn_overlap": {"knn_overlap_fraction": knn_overlap_fraction}
    }

# 3. Margin-based hardness (Approximate Margin)
def compute_margin_complexity(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing margin complexity (GPU)...")

    X_array = X.to_cupy().astype(cp.float32)
    y_array = y.to_cupy().astype(cp.int32)

    clf = LinearSVC(
        C=1.0,
        max_iter=100_000,
        tol=1e-4,
        fit_intercept=True,
        verbose=False
    )

    clf.fit(X_array, y_array)

    decision_function = clf.decision_function(X_array)

    if decision_function.ndim > 1 and decision_function.shape[1] > 1:
        margins = cp.min(cp.abs(decision_function), axis=1)
    else:
        margins = cp.abs(decision_function)

    return get_stats(margins)

# 4. Intrinsic Dimensionality (PCA variance ratio)
def compute_intrinsic_dimensionality(X: cudf.DataFrame, threshold: float = 0.95) -> dict:
    tqdm.write(f"[{now()}] Computing intrinsic dimensionality (GPU)...")

    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)

    pca = PCA(n_components=max_components)
    pca.fit(X)

    variance_ratio_cp = pca.explained_variance_ratio_.to_cupy()
    cumulative_variance = cp.cumsum(variance_ratio_cp)

    try:
        n_components = int(cp.argmax(cumulative_variance >= threshold)) + 1
    except Exception:
        n_components = len(cumulative_variance)

    intrinsic_dim_percent = n_components / n_features

    return {
        f"n_components_{threshold * 100:.0f}%": n_components,
        "intrinsic_dimensionality_percent": intrinsic_dim_percent,
        "total_features": n_features,
        "max_explained_variance": float(cumulative_variance[-1])
    }

# 5. Class Imbalance
def compute_class_imbalance(y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing class imbalance (GPU)...")

    class_counts = y.value_counts(sort=False)
    probs = class_counts.values / class_counts.values.sum()
    n_classes = len(class_counts)

    entropy = -cp.sum(probs * cp.log(probs))
    normalized_entropy = float(entropy / cp.log(n_classes)) if n_classes > 1 else 1.0

    imbalance_ratio = float(cp.max(class_counts.values) / cp.min(class_counts.values)) if n_classes > 1 else 1.0

    return {
        "n_classes": n_classes,
        "imbalance_ratio": imbalance_ratio,
        "normalized_entropy": normalized_entropy
    }

# 7. Class Centroid Distance
def compute_class_centroid_distance(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing class centroid distance (GPU)...")

    X_labeled = X.copy()
    X_labeled["label"] = y

    # Group by label to get centroids
    centroids = X_labeled.groupby("label").mean().drop(columns=["label"], errors="ignore").to_cupy()

    # Compute pairwise Euclidean distances
    diff = centroids[:, None, :] - centroids[None, :, :]
    dists = cp.sqrt(cp.sum(diff**2, axis=2))

    # Extract upper triangle without diagonal
    triu_vals = dists[cp.triu_indices(centroids.shape[0], k=1)]
    mean_distance = float(cp.mean(triu_vals)) if len(triu_vals) > 0 else 0.0

    return {"class_centroid_mean_distance": mean_distance}

# 8. Global Clusterability (Silhouette Score)
def compute_clusterability(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing clusterability (GPU)...")
    return {"silhouette_score": float(silhouette_score(X, y))}

# Master function to compute all complexity measures
def compute_all_complexity_measures(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Starting full complexity computation...")

    def safe_call(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tqdm.write(f"[{now()}] Error in {func.__name__}: {e}")
            return {}

    results = {
        'anova_f': safe_call(compute_anova_f_complexity, X, y),
        'mutual_info': safe_call(compute_mutual_info_complexity, X, y),
        'margin': safe_call(compute_margin_complexity, X, y),
        'intrinsic_dimensionality': safe_call(compute_intrinsic_dimensionality, X),
        'class_imbalance': safe_call(compute_class_imbalance, y),
        'centroid_distance': safe_call(compute_class_centroid_distance, X, y),
        'clusterability': safe_call(compute_clusterability, X, y),
    }

    # Merge KDN + Nearest Enemy + KNN Overlap
    merged_knn = safe_call(compute_local_overlap_and_knn_metrics, X, y)
    results['kdn'] = merged_knn.get('kdn', {})
    results['nearest_enemy'] = merged_knn.get('nearest_enemy', {})
    results['knn_overlap'] = merged_knn.get('knn_overlap', {})

    return results

def flatten_metrics_dict(metrics_dict: dict) -> pd.DataFrame:
    flat_dict = {}
    for top_key, subdict in metrics_dict.items():
        if isinstance(subdict, dict):
            for sub_key, value in subdict.items():
                flat_key = f"{top_key}_{sub_key}"
                flat_dict[flat_key] = value
        else:
            flat_dict[top_key] = subdict  # in case it's not nested
    return pd.DataFrame([flat_dict])

def compute_composite_difficulty_from_dict(metrics_dict: dict) -> pd.DataFrame:
    # Metric groups
    metric_groups = {
        "Feature_Relevance": [
            "anova_f_mean", 
            "mutual_info_mean"
        ],
        "Local_Overlap": [
            "kdn_mean", 
            "knn_overlap_knn_overlap_fraction"
        ],
        "Boundary_Hardness": [
            "margin_mean", 
            "nearest_enemy_mean"
        ],
        "Global_Structure": [
            "intrinsic_dimensionality_intrinsic_dimensionality_percent", 
            "clusterability_silhouette_score"
        ],
        "Class_Distribution_Separation": [
            "centroid_distance_class_centroid_mean_distance", 
            "class_imbalance_normalized_entropy"
        ]
    }

    # Metrics where higher = easier → invert so that higher = harder
    metrics_to_invert = [
        "anova_f_mean",
        "mutual_info_mean",
        "margin_mean",
        "nearest_enemy_mean",
        "clusterability_silhouette_score",
        "class_imbalance_normalized_entropy"
    ]

    # Flattened DataFrame with one row
    df = flatten_metrics_dict(metrics_dict)

    # Step 1: Invert selected metrics
    for metric in metrics_to_invert:
        if metric in df.columns:
            df[metric] = -df[metric]

    # Step 2: Normalize selected metrics
    scaler = MinMaxScaler()
    flatten_metrics = [
        metric
        for group in metric_groups.values()
        for metric in group
        if metric in df.columns
    ]
    if flatten_metrics:
        df[flatten_metrics] = scaler.fit_transform(df[flatten_metrics])

    # Step 3: Compute group difficulties (mean of normalized group metrics)
    for group_name, metric_list in metric_groups.items():
        available_metrics = [m for m in metric_list if m in df.columns]
        df[f"{group_name}_difficulty"] = (
            df[available_metrics].mean(axis=1) if available_metrics else np.nan
        )

    # Step 4: Compute overall difficulty (mean of group scores)
    group_cols = [f"{name}_difficulty" for name in metric_groups]
    df["overall_difficulty"] = df[group_cols].mean(axis=1)

    return df

# PYTHONPATH=. python modules/preprocessing/complexity_gpu_v2.py
if __name__ == "__main__":

    INPUT_FOLDER = '2025-06-28/Input_Multiclass'
    OUTPUT_FOLDER = '2025-06-28/Output_Multiclass'
    ONE_GB = 1 * 1024 * 1024 * 1024  # 1 GB
    SAMPLE_FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    SKIP = False

    candidate_files = sorted(
        list(Path(INPUT_FOLDER).rglob("*.parquet")),
        key=lambda p: os.path.getsize(p)
    )

    for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):
        sample_frac_suffix = str(int(100 * sample_frac)).zfill(3)

        for src_path in tqdm(candidate_files, desc='Candidate', leave=False):
            try:
                if not src_path.is_file():
                    continue

                abs_path = str(src_path.resolve())
                dst_path = Path(abs_path
                                .replace(INPUT_FOLDER, f'{OUTPUT_FOLDER}/{sample_frac_suffix}_pct')
                                .replace('.parquet', f'_complexity.json'))
                src_path_short = '/'.join(str(src_path).split('/')[-2:])
                os.makedirs(dst_path.parent, exist_ok=True)

                if SKIP and src_path.stat().st_size > ONE_GB:
                    tqdm.write(f"Skipping {src_path}; dataset is over 1GB.")
                    continue
                elif SKIP and dst_path.exists():
                    tqdm.write(f"Skipping {dst_path}; complexity JSON already exists.")
                    continue

                tqdm.write(f'[{now()}] DS: {src_path_short:<80} | PROCESSING @ FRAC={sample_frac:.2f}')

                # Read and clean with cuDF
                df_gpu = cudf.read_parquet(src_path)

                # Restore dtypes from metadata
                metadata_path = abs_path.replace('.parquet', '.json')
                with open(metadata_path, 'r', encoding='utf-8') as fp:
                    metadata = json.load(fp)
                for col, dtype in metadata['dtypes'].items():
                    df_gpu[col] = df_gpu[col].astype(dtype)

                # Type normalization
                for col in df_gpu.select_dtypes(['int8', 'int16', 'int64']).columns:
                    df_gpu[col] = df_gpu[col].astype('int32')
                for col in df_gpu.select_dtypes(['float16', 'float64']).columns:
                    df_gpu[col] = df_gpu[col].astype('float32')

                # Factorize categorical columns
                label_mappings = {}
                for col in df_gpu.select_dtypes(include=["category"]).columns:
                    df_gpu[col], mapping = df_gpu[col].factorize()
                    label_mappings[col] = mapping.to_pandas().tolist()

                # Drop NAs and duplicates
                df_gpu = df_gpu.replace([cp.inf, -cp.inf], cp.nan)
                df_gpu = df_gpu.dropna(axis=1, how='all')
                df_gpu = df_gpu.dropna(axis=0, how='any')
                df_gpu = df_gpu.drop_duplicates()

                # Target column and dtypes pre-checks
                assert 'label' in df_gpu.columns, "'label' column is missing"
                remaining_object_cols = df_gpu.select_dtypes(include=["object", "str", "category"]).columns
                assert len(remaining_object_cols) == 0, f"Unencoded categorical columns remain: {list(remaining_object_cols)}"

                # Save shapes before preprocessing
                shape_before = df_gpu.shape
                n_classes_before = df_gpu['label'].nunique()

                # Stratified sample (CPU fallback)
                df_gpu = stratified_sample_with_min(
                    df=df_gpu,
                    stratify_col='label',
                    max_total_samples=int(sample_frac * len(df_gpu)),
                    min_samples_per_class=max(10, min(df_gpu['label'].value_counts().values_host))
                )

                # Post-checks
                shape_after = df_gpu.shape
                n_classes_after = df_gpu['label'].nunique()
                value_counts = df_gpu['label'].value_counts()
                values_counts_mapped = {
                    label_mappings['label'][k]: value_counts[k] for k in value_counts.index.values_host
                }
                assert n_classes_after > 2, (
                    f"Sampled dataset has 2 or less labels: {values_counts_mapped}"
                )
                assert n_classes_after == n_classes_before, (
                    f"Class count changed during sampling: before={n_classes_before}, after={n_classes_after}"
                )
                assert shape_after[1] == shape_before[1], (
                    f"Sampled dataset column mismatch: before={shape_before[1]}, after={shape_after[1]}"
                )

                tqdm.write(f'[{now()}] DS: {src_path_short:<80} | SHAPE_0: {str(shape_before):<16} | SHAPE_1: {str(shape_after):<16}')

                X, y = df_gpu.drop(columns=['label']), df_gpu['label']

                # Compute metrics
                metrics = compute_all_complexity_measures(X, y)
                metrics['label_mappings'] = label_mappings
                metrics['errors'] = []

                expected_keys = [
                    'anova_f', 'mutual_info', 'kdn', 'nearest_enemy',
                    'margin', 'intrinsic_dimensionality', 'class_imbalance',
                    'knn_overlap', 'centroid_distance', 'clusterability'
                ]

                for key in expected_keys:
                    if key in metrics:
                        for subkey, value in metrics[key].items():
                            if not isinstance(value, (int, float)):
                                metrics['errors'].append({
                                    'key': key, 'subkey': subkey,
                                    'message': f'{value} is not int/float'
                                })
                    else:
                        metrics['errors'].append({
                            'key': key,
                            'message': 'metric is missing'
                        })

                with open(dst_path, 'w') as fp:
                    json.dump(metrics, fp, indent=4)

                debug_json_path = Path(
                    str(dst_path)
                        .replace(INPUT_FOLDER, f'{OUTPUT_FOLDER}/{sample_frac_suffix}_pct')
                        .replace('_complexity.json', '_complexity.error' if metrics['errors'] else '_complexity.success')
                )
                debug_json_path.touch()

                tqdm.write(f'[{now()}] DS: {src_path_short:<80} | METRICS: {metrics}\n')

                del df_gpu
                gc.collect()

            except Exception as e:
                tqdm.write(f"[{now()}] Error in {src_path}: {e}")
