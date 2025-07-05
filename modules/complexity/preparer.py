import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Tuple


def now():
    now = datetime.now()
    yyyymmdd_hhmmss_part = now.strftime('%Y-%m-%d %H:%M:%S')
    ms_part = f'{int(now.microsecond / 1000):03d}'
    return f'{yyyymmdd_hhmmss_part},{ms_part}'


def stratified_sample_with_min(
    df: pd.DataFrame,
    stratify_col: str,
    max_total_samples: int,
    min_samples_per_class: int
) -> pd.DataFrame:
    # Step 0: Oversample minority classes if needed
    class_counts = df[stratify_col].value_counts()
    minority_classes = class_counts[class_counts < min_samples_per_class]

    if len(minority_classes) > 0:
        oversampled = []
        for cls in minority_classes.index:
            group = df[df[stratify_col] == cls]
            samples = group.sample(n=min_samples_per_class, replace=True, random_state=42)
            oversampled.append(samples)
        df_balanced = pd.concat([df] + oversampled, ignore_index=True)
    else:
        df_balanced = df

    # If total size is already under the threshold, return full balanced set
    if len(df_balanced) <= max_total_samples:
        return df_balanced

    # Step 1: Recompute proportions
    class_proportions = df_balanced[stratify_col].value_counts(normalize=True)

    # Step 2: Initial per-class sample counts
    sample_counts = (class_proportions * max_total_samples).round().astype(int)

    # Step 3: Enforce min_samples_per_class
    sample_counts[sample_counts < min_samples_per_class] = min_samples_per_class

    # Step 4: Re-normalize if total > max_total_samples
    total = sample_counts.sum()
    if total > max_total_samples:
        scale = max_total_samples / total
        sample_counts = (sample_counts * scale).round().astype(int)
        sample_counts[sample_counts < min_samples_per_class] = min_samples_per_class

    # Step 5: Final stratified sampling
    sampled = []
    for cls, count in sample_counts.items():
        group = df_balanced[df_balanced[stratify_col] == cls]
        count = min(count, len(group))
        sampled.append(group.sample(n=count, replace=False, random_state=42))

    return pd.concat(sampled, ignore_index=True)


def preprocess_factorized(
    X_raw: pd.DataFrame,
    y: pd.Series = None,
    numeric_strategy: str = "scale",
    cat_encoding: str = "frequency",
    reduce_dim: bool = True,
    cum_var_threshold: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess mixed-type Pandas dataframe for FAISS use.
    - Scales numeric features
    - Frequency encodes low-cardinality categoricals (nunique < 200)
    - Optionally applies PCA to reduce dimensionality (preserving cum_var_threshold variance)
    - Returns a float32 NumPy array
    """
    # Step 0: Identify numeric and categorical columns
    num_cols = X_raw.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = [col for col in num_cols if X_raw[col].nunique() < 200]
    num_cols = [col for col in num_cols if col not in cat_cols]

    X_num = X_raw[num_cols].astype("float32")
    X_cat = X_raw[cat_cols].astype("int32")

    # Step 1: Scale numeric features
    if numeric_strategy == "scale" and not X_num.empty:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        X_num_scaled = X_num.to_numpy(dtype=np.float32)

    # Step 2: Frequency encode categoricals
    if cat_encoding == "frequency" and not X_cat.empty:
        X_cat_encoded = pd.DataFrame(index=X_cat.index)
        for col in X_cat.columns:
            freq_map = X_cat[col].value_counts(normalize=True).to_dict()
            X_cat_encoded[col + "_freq"] = X_cat[col].map(freq_map).astype("float32")
        X_cat_encoded = X_cat_encoded.to_numpy(dtype=np.float32)
    elif not X_cat.empty:
        raise ValueError(f"Unsupported categorical encoding: {cat_encoding}")
    else:
        X_cat_encoded = np.empty((len(X_raw), 0), dtype=np.float32)

    # Step 3: Combine numeric + categorical
    X_all = np.hstack([X_num_scaled, X_cat_encoded])

    # Step 4: Dimensionality reduction
    if reduce_dim and X_all.shape[1] > 0:
        pca = PCA(n_components=min(X_all.shape[0], X_all.shape[1]), svd_solver="full")
        X_pca_all = pca.fit_transform(X_all)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = int(np.argmax(cum_var >= cum_var_threshold)) + 1
        X_final = X_pca_all[:, :n_components_95].astype(np.float32)
    else:
        X_final = X_all.astype(np.float32)

    return X_final, y.astype(np.int32).to_numpy()


def safe_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        tqdm.write(f"[{now()}] Error in {func.__name__}: {e}")
        return {}


# PYTHONPATH=. python modules/complexity/preparer.py
if __name__ == "__main__":

    TARGET_COL = 'label'
    INPUT_FOLDER = '2025-07-05/Input_Multiclass'
    OUTPUT_FOLDER = '2025-07-05/Output_Multiclass'
    SAMPLE_FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    SKIP_IF_EXISTS = False

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
                                .replace('.parquet', f'.npz'))
                src_path_short = '/'.join(str(src_path).split('/')[-2:])
                dst_path_short = '/'.join(str(dst_path).split('/')[-3:])
                os.makedirs(dst_path.parent, exist_ok=True)

                if SKIP_IF_EXISTS and dst_path.exists():
                    tqdm.write(f"Skipping {dst_path}; npz already exists.")
                    continue

                tqdm.write(f'[{now()}] DS: {src_path_short:<80} | PROCESSING @ FRAC={sample_frac:.2f}')

                # Read and clean with cuDF
                df = pd.read_parquet(src_path)

                # Restore dtypes from metadata
                metadata_path = abs_path.replace('.parquet', '.json')
                with open(metadata_path, 'r', encoding='utf-8') as fp:
                    metadata = json.load(fp)
                for col, dtype in metadata['dtypes'].items():
                    df[col] = df[col].astype(dtype)

                # Type normalization
                for col in df.select_dtypes(['int8', 'int16', 'int64']).columns:
                    df[col] = df[col].astype('int32')
                for col in df.select_dtypes(['float16', 'float64']).columns:
                    df[col] = df[col].astype('float32')

                # Factorize categorical columns
                label_mappings = {}
                for col in df.select_dtypes(include=["category"]).columns:
                    df[col], mapping = df[col].factorize()
                    label_mappings[col] = mapping.tolist()

                # Drop NAs and duplicates
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna(axis=1, how='all')
                df = df.dropna(axis=0, how='any')
                df = df.drop_duplicates()

                # Target column and dtypes pre-checks
                assert TARGET_COL in df.columns, "TARGET_COL column is missing"
                remaining_object_cols = df.select_dtypes(include=["object", "category"]).columns
                assert len(remaining_object_cols) == 0, f"Unencoded categorical columns remain: {list(remaining_object_cols)}"

                # Save shapes before preprocessing
                shape_before = df.shape
                n_classes_before = df[TARGET_COL].nunique()

                # Stratified sample
                df = stratified_sample_with_min(
                    df=df,
                    stratify_col=TARGET_COL,
                    max_total_samples=int(sample_frac * len(df)),
                    min_samples_per_class=max(10, min(df[TARGET_COL].value_counts()))
                )
                
                # Further preprocessing for metrics
                X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
                X, y = preprocess_factorized(X, y, reduce_dim=False)

                assert np.isnan(X).sum() == 0, 'NaNs found in X'
                assert np.isnan(y).sum() == 0, 'NaNs found in y'

                np.savez(dst_path, X=X, y=y)
                npz_size = dst_path.stat().st_size / 1024 / 1024

                tqdm.write(f'[{now()}] DS: {dst_path_short:<80} | ___DONE___ @ FRAC={sample_frac:.2f} | SIZE={npz_size:.2f}MB')

            except Exception as e:
                tqdm.write(f"[{now()}] Error in {src_path}: {e}")
