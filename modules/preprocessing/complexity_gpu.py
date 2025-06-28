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
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
from cuml.svm import LinearSVC
from sklearn.feature_selection import f_classif, mutual_info_classif
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

def stratified_sample_with_min(df, stratify_col, max_total_samples, min_samples_per_class):
    # Step 0: Handle case where dataset is smaller than max_total_samples
    if len(df) <= max_total_samples:
        # Ensure minimum samples per class via oversampling
        class_counts = df[stratify_col].value_counts()
        oversampled = [
            df[df[stratify_col] == cls].sample(n=min_samples_per_class, replace=True)
            for cls in class_counts[class_counts < min_samples_per_class].index
        ]
        df_balanced = pd.concat([df] + oversampled, ignore_index=True) if oversampled else df
        return df_balanced.reset_index(drop=True)

    # Step 1: Ensure minimum samples per class (oversample if needed)
    class_counts = df[stratify_col].value_counts()
    oversampled = [
        df[df[stratify_col] == cls].sample(n=min_samples_per_class, replace=True)
        for cls in class_counts[class_counts < min_samples_per_class].index
    ]
    df_balanced = pd.concat([df] + oversampled, ignore_index=True) if oversampled else df

    # Step 2: Recompute class proportions
    class_proportions = df_balanced[stratify_col].value_counts(normalize=True)

    # Step 3: Compute how many samples to draw per class
    class_sample_counts = (class_proportions * max_total_samples).round().astype(int)

    # Step 4: Enforce min_samples_per_class
    class_sample_counts[class_sample_counts < min_samples_per_class] = min_samples_per_class

    # Step 5: Re-normalize if total exceeds max_total_samples
    total = class_sample_counts.sum()
    if total > max_total_samples:
        scaling_factor = max_total_samples / total
        class_sample_counts = (class_sample_counts * scaling_factor).round().astype(int)
        class_sample_counts[class_sample_counts < min_samples_per_class] = min_samples_per_class

    # Final sampling
    sampled_df = pd.concat([
        df_balanced[df_balanced[stratify_col] == cls].sample(
            n=min(count, len(df_balanced[df_balanced[stratify_col] == cls])),
            replace=False
        )
        for cls, count in class_sample_counts.items()
    ], ignore_index=True)

    return sampled_df.reset_index(drop=True)

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

    # Convert to CuPy arrays
    X_array = X.to_cupy()
    y_array = y.to_cupy()

    clf = LinearSVC(
        C=1.0,
        max_iter=20000,
        tol=1e-4,
        fit_intercept=True,
        verbose=False
    )

    clf.fit(X_array, y_array)

    # Get distances to hyperplane
    decision_function = clf.decision_function(X_array)

    # Handle multiclass and binary
    if decision_function.ndim > 1 and decision_function.shape[1] > 1:
        margins = cp.min(cp.abs(decision_function), axis=1)
    else:
        margins = cp.abs(decision_function)

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
    # Convert inputs to cuDF for GPU processing
    X_cu = cudf.DataFrame.from_pandas(X_pd) if isinstance(X_pd, pd.DataFrame) else X_pd
    y_cu = cudf.Series.from_pandas(y_pd) if isinstance(y_pd, pd.Series) else y_pd
    # X_cu, y_cu = smart_categorical_encode(X_cu, y_cu)

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
        'nearest_enemy': safe_call(compute_nearest_enemy_complexity, X_cu, y_cu),
        'margin': safe_call(compute_margin_complexity, X_cu, y_cu),
        'intrinsic_dimensionality': safe_call(compute_intrinsic_dimensionality, X_cu),
        'class_imbalance': safe_call(compute_class_imbalance, y_cu),
        'knn_overlap': safe_call(compute_knn_overlap, X_cu, y_cu),
        'centroid_distance': safe_call(compute_class_centroid_distance, X_cu, y_cu),
        'clusterability': safe_call(compute_clusterability, X_cu, y_cu),
    }

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

    # Corrected metric groups
    metric_groups = {
        "Feature_Relevance": ["anova_f_mean", "mutual_info_mean"],
        "Local_Overlap": ["kdn_mean", "knn_overlap_knn_overlap_fraction"],
        "Boundary_Hardness": ["margin_mean", "nearest_enemy_mean"],
        "Global_Structure": ["intrinsic_dimensionality_intrinsic_dimensionality_percent", "clusterability_silhouette_score"],
        "Class_Distribution_Separation": ["centroid_distance_class_centroid_mean_distance", "class_imbalance_normalized_entropy"]
    }

    # Corrected flip list
    metrics_to_invert = [
        "anova_f_mean",
        "mutual_info_mean",
        "margin_mean",
        "nearest_enemy_mean",
        "clusterability_silhouette_score",
        "class_imbalance_normalized_entropy"
    ]

    df = flatten_metrics_dict(metrics_dict)

    # Step 1: Flip necessary metrics
    for metric in metrics_to_invert:
        if metric in df.columns:
            df[metric] = -df[metric]

    # Step 2: Normalize
    scaler = MinMaxScaler()
    flatten_metrics = [
        metric                                 # Final flattened metric name (e.g., "anova_f_mean")
        for metrics in metric_groups.values()   # Each metric list from each group
        for metric in metrics                   # Each metric name in that list
        if metric in df.columns                 # Only include if it exists in the current DataFrame
    ]
    df[flatten_metrics] = scaler.fit_transform(df[flatten_metrics])

    # Step 3: Group difficulty
    for group_name, metric_list in metric_groups.items():
        available_metrics = [m for m in metric_list if m in df.columns]
        if available_metrics:
            df[group_name + "_difficulty"] = df[available_metrics].mean(axis=1)
        else:
            df[group_name + "_difficulty"] = np.nan

    # Step 4: Overall difficulty
    group_difficulty_cols = [g + "_difficulty" for g in metric_groups.keys()]
    df["overall_difficulty"] = df[group_difficulty_cols].mean(axis=1)

    return df

# PYTHONPATH=. python modules/preprocessing/complexity_gpu_v2.py
if __name__ == "__main__":

    INPUT_FOLDER = '2025-06-26-b/Input_Multiclass'
    OUTPUT_FOLDER = '2025-06-26-b/Output_Multiclass'
    ONE_GB = 1 * 1024 * 1024 * 1024 # 1 GB in bytes
    SAMPLE_FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    SKIP = True

    candidate_files = list(Path(INPUT_FOLDER).rglob("*.parquet"))

    candidate_files = sorted(candidate_files, key=lambda p: os.path.getsize(p))

    for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):

        sample_frac_suffix = str(int(100 * sample_frac)).zfill(3)

        for src_path in tqdm(candidate_files, desc='Candidate', leave=False):

            try:

                # Checks
                if src_path.is_file():
                    abs_path = str(src_path.absolute().resolve())
                    dst_path = Path(abs_path
                                    .replace(INPUT_FOLDER, f'{OUTPUT_FOLDER}/{sample_frac_suffix}_pct')
                                    .replace('.parquet', f'_complexity.json'))
                    src_path_short = '/'.join(str(src_path).split('/')[-2:])
                    os.makedirs(dst_path.parent, exist_ok=True)

                    if SKIP and src_path.stat().st_size > ONE_GB:
                        tqdm.write(f"Skipping {src_path}; dataset is over 1GB.")
                        # exit(0)
                        continue
                    elif SKIP and dst_path.exists() and dst_path.is_file():
                        # dst_path.unlink()
                        tqdm.write(f"Skipping {dst_path}; complexity JSON already exists.")
                        # exit(0)
                        continue
                    else:
                        tqdm.write(f'[{now()}] DS: {src_path_short:<80} | PROCESSING @ FRAC={sample_frac:.2f}')

                    # DF (CPU)
                    df_cpu = pd.read_parquet(src_path)
                    shape_before = df_cpu.shape

                    # Cast integer types to int32
                    df_cpu = df_cpu.astype({col: 'int32' for col in df_cpu.select_dtypes(['int8', 'int16', 'int64']).columns})
                    # Cast float types to float32
                    df_cpu = df_cpu.astype({col: 'float32' for col in df_cpu.select_dtypes(['float16', 'float64']).columns})
                    # Factorize categorical columns into int32 codes
                    for col in df_cpu.select_dtypes(include='category').columns:
                        df_cpu[col] = pd.factorize(df_cpu[col])[0].astype('int32')

                    df_cpu = df_cpu.replace([np.inf, -np.inf], np.nan)
                    df_cpu = df_cpu.dropna(axis='columns', how='all')
                    df_cpu = df_cpu.dropna(axis='index', how='any')

                    df_cpu['label'], label_mapping = df_cpu['label'].factorize()
                    n_classes_orig = df_cpu['label'].nunique()
                    df_cpu = stratified_sample_with_min(
                        df=df_cpu,
                        stratify_col='label',
                        max_total_samples=int(sample_frac * df_cpu.shape[0]),
                        min_samples_per_class=int(1.0 / sample_frac)
                    )
                    assert df_cpu['label'].nunique() == n_classes_orig, "Some classes were lost during stratification."
                    label_mapping_dict = {str(label_mapping[i]): int(i) for i in range(len(label_mapping))}
                    shape_after = df_cpu.shape

                    # DF (GPU)
                    df_gpu = cudf.from_pandas(df_cpu)
                    assert df_gpu['label'].nunique() == n_classes_orig, "Some classes were lost during CPU -> GPU."
                    del df_cpu
                    gc.collect()

                    tqdm.write(f'[{now()}] DS: {src_path_short:<80} | SHAPE_0: {str(shape_before):<16} | SHAPE_1: {str(shape_after):<16}')
                    X, y = df_gpu.drop(columns=['label']), df_gpu['label']
                    # X, y = smart_categorical_encode(X, y)

                    # Metrics
                    metrics = compute_all_complexity_measures(X, y)                    
                    # metrics['mappings'] = label_mapping_dict
                    metrics['errors'] = []

                    expected_keys = [
                        'anova_f', 'mutual_info', 'kdn', 'nearest_enemy',
                        'margin', 'intrinsic_dimensionality', 'class_imbalance',
                        'knn_overlap', 'centroid_distance', 'clusterability'
                    ]

                    for key in expected_keys:
                        if key in metrics.keys():
                            for subkey, value in metrics[key].items():
                                if not isinstance(value, (int, float)):
                                    metrics['errors'].append({
                                        'key': key,
                                        'subkey':subkey, 
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
                        str(dst_path.absolute().resolve())
                            .replace(INPUT_FOLDER, f'{OUTPUT_FOLDER}/{sample_frac_suffix}_pct')
                            .replace('_complexity.json', '_complexity.error' if len(metrics['errors']) > 0 else '_complexity.success')
                    )
                    Path(debug_json_path).touch()

                    tqdm.write(f'[{now()}] DS: {src_path_short:<80} | METRICS: {metrics}\n')
                    del df_gpu
                    gc.collect()

                    # TODO aggregate

                    # import json
                    # from pathlib import Path
                    # import pandas as pd

                    # json_files = list(Path("Output_Multiclass").rglob("*_complexity.json"))

                    # records = []
                    # for file in json_files:
                    #     with open(file) as f:
                    #         metrics = json.load(f)
                    #     flattened = flatten_metrics_dict(metrics)
                    #     flattened["dataset_name"] = file.stem.replace("_complexity", "")
                    #     records.append(flattened)

                    # df_metrics = pd.concat(records, ignore_index=True)

                    # df_difficulty = compute_composite_difficulty(df_metrics)

                    # df_difficulty.sort_values("overall_difficulty", ascending=False)

            except Exception as e:

                tqdm.write(f"[{now()}] Error in {src_path}: {e}")
