import json
import math
import os
import warnings
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
from cuml.decomposition import PCA
from cuml.linear_model import LogisticRegression
from cuml.svm import LinearSVC
from tqdm import tqdm

from modules.complexity.preparer import now, safe_call

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*DataFrameGroupBy\\.apply operated on the grouping columns.*" )
warnings.filterwarnings("ignore", category=FutureWarning, message=".*default of observed=False is deprecated.*")

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

################################################################################

# 1.1 Feature Relevance - ANOVA F Statistic
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


# 1.2 Feature Relevance - Mutual Information
def compute_mutual_info_complexity(X: cudf.DataFrame, y: cudf.Series, max_bins: int = 64) -> dict:
    tqdm.write(f"[{now()}] Computing mutual information complexity (GPU)...")

    X_cp = X.to_cupy().astype(cp.float32)
    y_cp = y.to_cupy().astype(cp.int32)
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

################################################################################

# 2.1 Local Overlap: PCA Class Centroid Distance
def compute_pca_centroid_score(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing PCA centroid separation (GPU)...")

    n_samples, n_features = X.shape

    pca = PCA(n_components=n_features, output_type="cupy")
    X_pca = pca.fit_transform(X)

    y_cp = y.to_cupy().astype(cp.int32)
    classes = cp.unique(y_cp)

    # Compute centroids per class in PCA space
    centroids = cp.vstack([
        X_pca[y_cp == c].mean(axis=0) for c in classes
    ])

    # Pairwise distances between centroids
    dists = cp.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    upper_tri = cp.triu_indices(len(classes), k=1)
    score = cp.mean(dists[upper_tri])

    return {"pca_centroid_score": float(score)}

# 2.2 Local Overlap - Mahalanobis Class Distance 
def compute_mahalanobis_class_distance(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing Mahalanobis class distance (GPU)...")

    X_cp = X.to_cupy().astype(cp.float32)
    y_cp = y.to_cupy().astype(cp.int32)
    classes = cp.unique(y_cp)

    distances = []

    for c in classes:
        class_mask = (y_cp == c)
        X_class = X_cp[class_mask]
        mean_c = cp.mean(X_class, axis=0)
        cov_c = cp.cov(X_class, rowvar=False)
        
        # Regularize covariance to avoid singularity
        cov_c += cp.eye(cov_c.shape[0]) * 1e-6
        
        inv_cov_c = cp.linalg.inv(cov_c)
        diff = X_class - mean_c
        dists = cp.sqrt(cp.sum(diff @ inv_cov_c * diff, axis=1))
        distances.append(dists)

    all_dists = cp.concatenate(distances)
    return get_stats(all_dists)

################################################################################

# 3.1: Boundary Hardness: Linear SVM Margin
def compute_linear_svm_margin(X: cudf.DataFrame, y: cudf.Series) -> dict:
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


# 3.2: Boundary Hardness: Entropy of Class Probabilities
def compute_class_proba_entropy(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing entropy-based boundary complexity (GPU)...")

    X_cp = X.to_cupy().astype(cp.float32)
    y_cp = y.to_cupy().astype(cp.int32)

    clf = LogisticRegression(
        C=1.0,
        max_iter=100_000,
        tol=1e-4,
        fit_intercept=True,
        verbose=False
    )
    clf.fit(X_cp, y_cp)

    probs = clf.predict_proba(X_cp)
    log_probs = cp.log(probs + 1e-10)  # for numerical stability
    entropy = -cp.sum(probs * log_probs, axis=1)

    return get_stats(entropy)

################################################################################

# 4.1 Global Structure - PCA Intrinsic Dimensionality
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


# 4.2 Global Structure - Global Clusterability (Calinski-Harabasz Index)
def compute_calinski_harabasz_score(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing Calinski-Harabasz Index (GPU)...")

    X_cp = X.to_cupy()
    y_cp = y.to_cupy()
    classes = cp.unique(y_cp)

    n_samples, n_features = X_cp.shape
    k = len(classes)

    overall_mean = cp.mean(X_cp, axis=0)

    # Between-class scatter
    ss_between = 0.0
    for c in classes:
        class_mask = y_cp == c
        X_c = X_cp[class_mask]
        n_c = X_c.shape[0]
        class_mean = cp.mean(X_c, axis=0)
        ss_between += n_c * cp.sum((class_mean - overall_mean) ** 2)

    # Within-class scatter
    ss_within = 0.0
    for c in classes:
        class_mask = y_cp == c
        X_c = X_cp[class_mask]
        class_mean = cp.mean(X_c, axis=0)
        ss_within += cp.sum((X_c - class_mean) ** 2)

    ch_index = (ss_between / (k - 1)) / (ss_within / (n_samples - k))

    return {"calinski_harabasz_score": float(ch_index)}

################################################################################

# 5.1 Class Distribution & Separation: Class Confusion Entropy
def compute_confusion_entropy(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Computing confusion entropy (GPU)...")

    clf = LogisticRegression(
        C=1.0,
        max_iter=100_000,
        tol=1e-4,
        fit_intercept=True,
        verbose=False
    )
    clf.fit(X, y)
    y_pred = clf.predict(X).to_cupy().astype(cp.int32)
    y_true = y.to_cupy().astype(cp.int32)

    n_classes = len(cp.unique(y_true))
    conf_matrix = cp.zeros((n_classes, n_classes), dtype=cp.int32)
    flat_indices = y_true * n_classes + y_pred
    counts = cp.bincount(flat_indices, minlength=n_classes * n_classes)
    conf_matrix = counts.reshape((n_classes, n_classes))

    # cp.atomic.add(conf_matrix, (y_true, y_pred), 1)

    # for i in range(len(y_true)):
    #     conf_matrix[y_true[i], y_pred[i]] += 1

    row_sums = cp.sum(conf_matrix, axis=1, keepdims=True)
    row_probs = conf_matrix / (row_sums + 1e-10)

    logp = cp.log(row_probs + 1e-10)
    entropy = -cp.sum(row_probs * logp, axis=1)
    mean_entropy = float(cp.mean(entropy))

    return {"confusion_entropy": mean_entropy}


# 5.2 Class Distribution & Separation: Class Imbalance
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

################################################################################

def compute_cuml_complexity_metrics(X: cudf.DataFrame, y: cudf.Series) -> dict:
    tqdm.write(f"[{now()}] Starting full complexity computation...")

    results = {
        # 1. Feature Relevance
        "anova_f": safe_call(compute_anova_f_complexity, X, y),
        "mutual_info": safe_call(compute_mutual_info_complexity, X, y),

        # 2. Local Overlap
        "pca_centroid_distance": safe_call(compute_pca_centroid_score, X, y),
        'mahalanobis_class_distance': safe_call(compute_mahalanobis_class_distance, X, y),

        # 3. Boundary Hardness
        "svm_margin": safe_call(compute_linear_svm_margin, X, y),
        "class_proba_entropy": safe_call(compute_class_proba_entropy, X, y),

        # 4. Global Structure
        "intrinsic_dimensionality": safe_call(compute_intrinsic_dimensionality, X),
        "calinski_harabasz": safe_call(compute_calinski_harabasz_score, X, y),

        # 5. Class Distribution & Separation
        "class_confusion_entropy": safe_call(compute_confusion_entropy, X, y),
        "class_imbalance": safe_call(compute_class_imbalance, y),
    }

    return results

################################################################################

# PYTHONPATH=. python modules/complexity/calculator.py
# Requirements: python 3.12, cuda 12.x, rapids (cupy, cuml, cudf)
if __name__ == "__main__":

    TARGET_COL = 'label'
    BASE_FOLDER = '2025-07-05/Output_Multiclass'
    SAMPLE_FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    SKIP_IF_EXISTS = False

    for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):

        sample_frac_suffix = str(int(100 * sample_frac)).zfill(3)
        
        base_folder_full = Path(f'{BASE_FOLDER}/{sample_frac_suffix}_pct')

        candidate_files = sorted(list(base_folder_full.rglob("*.npz")), key=lambda p: os.path.getsize(p))
        
        for src_path in tqdm(candidate_files, desc='Candidate', leave=False):
            try:
                if not src_path.is_file():
                    continue

                abs_path = str(src_path.resolve())
                dst_path = Path(abs_path.replace('.npz', f'_complexity.json'))
                src_path_short = '/'.join(str(src_path).split('/')[-2:])
                src_size = src_path.stat().st_size / 1024 / 1024
                os.makedirs(dst_path.parent, exist_ok=True)

                if SKIP_IF_EXISTS and dst_path.exists():
                    tqdm.write(f"Skipping {dst_path}; complexity JSON already exists.")
                    continue

                tqdm.write(f'[{now()}] DS: {src_path_short:<80} | PROCESSING @ FRAC={sample_frac:.2f} | SIZE={src_size:.2f}MB')

                # Load data from .npz file
                data = np.load(src_path)
                X = cudf.DataFrame(data["X"])
                y = cudf.Series(data["y"])
                
                # X and y must have the same number of rows
                assert X.shape[0] == y.shape[0], "Mismatch between X and y lengths"

                # All columns in X must be floating point
                for col in X.columns:
                    assert np.issubdtype(X[col].dtype, np.floating), f"X[{col}] must be float"

                # y must be integer
                assert np.issubdtype(y.dtype, np.integer), "y must be integer type"

                # Compute metrics
                metrics = compute_cuml_complexity_metrics(X, y)
                metrics['errors'] = []

                # Define the list of expected top-level metric keys
                expected_keys = [
                    "anova_f", "mutual_info",                               # 1. Feature Relevance
                    "pca_centroid_distance", "mahalanobis_class_distance",  # 2. Local Overlap
                    "svm_margin", "class_proba_entropy",                    # 3. Boundary Hardness
                    "intrinsic_dimensionality", "calinski_harabasz",        # 4. Global Structure
                    "class_confusion_entropy", "class_imbalance"            # 5. Class Distribution & Separation
                ]

                # Validate each expected metric
                for key in expected_keys:
                    if key in metrics:
                        # Check that each submetric value under the key is a number
                        for subkey, value in metrics[key].items():
                            if not isinstance(value, (int, float)):
                                metrics['errors'].append({
                                    'key': key,
                                    'subkey': subkey,
                                    'message': f'{value} is not int/float'
                                })
                            elif isinstance(value, float) and math.isnan(value):
                                metrics['errors'].append({
                                    'key': key,
                                    'subkey': subkey,
                                    'message': 'value is NaN'
                                })
                    else:
                        # Report if the metric key is completely missing
                        metrics['errors'].append({
                            'key': key,
                            'message': 'metric is missing'
                        })

                # Write the updated metrics (including any errors) to a JSON file
                with open(dst_path, 'w') as fp:
                    json.dump(metrics, fp, indent=4)

                # Replace suffix based on whether errors are present
                debug_json_path = Path(
                    str(dst_path).replace(
                        "_complexity.json",
                        "_complexity.error" if metrics["errors"] else "_complexity.success"
                    )
                )
                debug_json_path.touch()

                tqdm.write(f'[{now()}] DS: {src_path_short:<80} | METRICS: {metrics}\n')

            except Exception as e:
                tqdm.write(f"[{now()}] Error in {src_path}: {e}")
