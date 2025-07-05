from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import json
import os
import pandas as pd

# Define metric groups
metric_groups = {
    "Feature_Relevance": [
        "anova_f_mean", 
        "mutual_info_mean"
    ],
    "Local_Overlap": [
        "pca_centroid_distance_pca_centroid_score", 
        "mahalanobis_class_distance_mean"
    ],
    "Boundary_Hardness": [
        "svm_margin_mean", 
        "class_proba_entropy_mean"
    ],
    "Global_Structure": [
        "intrinsic_dimensionality_intrinsic_dimensionality_percent", 
        "calinski_harabasz_calinski_harabasz_score"
    ],
    "Class_Distribution_Separation": [
        "class_confusion_entropy_confusion_entropy", 
        "class_imbalance_normalized_entropy"
    ]
}

def flatten_metrics_dict(metrics_dict: dict, dataset_id: str, keys_to_include=None) -> dict:
    flat = {"dataset_id": dataset_id}
    for top_key, subdict in metrics_dict.items():
        if isinstance(subdict, dict):
            for sub_key, value in subdict.items():
                flat_key = f"{top_key}_{sub_key}"
                if keys_to_include is None or flat_key in keys_to_include:
                    flat[flat_key] = value
        else:
            if keys_to_include is None or top_key in keys_to_include:
                flat[top_key] = subdict
    return flat

def compute_composite_difficulty_from_dict(metrics_dict: dict, dataset_id: str, min_metrics_per_group=1) -> pd.DataFrame | None:
    all_metrics = [m for group in metric_groups.values() for m in group]

    flat_dict = flatten_metrics_dict(metrics_dict, dataset_id, keys_to_include=all_metrics)
    df = pd.DataFrame([flat_dict])

    available_metrics = []
    missing_metrics = []

    for metric in all_metrics:
        if metric in df.columns and not pd.isna(df.loc[0, metric]):
            available_metrics.append(metric)
        else:
            missing_metrics.append(metric)

    if missing_metrics:
        print(f"[INFO] {dataset_id}: Missing metrics: {missing_metrics}")

    valid_groups = {}
    for group_name, metric_list in metric_groups.items():
        valid_metrics_in_group = [m for m in metric_list if m in available_metrics]
        if len(valid_metrics_in_group) >= min_metrics_per_group:
            valid_groups[group_name] = valid_metrics_in_group
        else:
            print(f"[WARN] {dataset_id}: Group '{group_name}' has only {len(valid_metrics_in_group)} valid metrics")

    if len(valid_groups) < 3:
        print(f"[SKIP] {dataset_id}: Only {len(valid_groups)} valid groups, need at least 3")
        return None

    # Invert selected metrics (where higher = easier)
    metrics_to_invert = {
        "anova_f_mean",
        "mutual_info_mean",
        "svm_margin_mean",
        "pca_centroid_score",
        "mahalanobis_class_distance_mean",
        "calinski_harabasz_calinski_harabasz_score",
        "class_imbalance_normalized_entropy"
    }

    for metric in metrics_to_invert:
        if metric in available_metrics:
            df[metric] = -df[metric]

    # print(f"[INFO] {dataset_id}: Skipping normalization for single dataset")

    for group_name, metric_list in valid_groups.items():
        df[f"{group_name}_difficulty"] = df[metric_list].mean(axis=1)

    group_cols = [f"{g}_difficulty" for g in valid_groups.keys()]
    df["overall_difficulty"] = df[group_cols].mean(axis=1)
    df["metrics_used"] = len(available_metrics)
    df["groups_used"] = len(valid_groups)

    return df

def compute_all_composite_difficulties(root_dir: str, suffix: str = "_complexity_cuml.json", min_metrics_per_group=1) -> pd.DataFrame:
    exclude_substrings = {
        "CICAPT_IIoT_Phase1_Macro_Multiclass",
        "CICAPT_IIoT_Phase1_Micro_Multiclass",
        "ToN_IoT_IoT_Motion_Light_Multiclass"
    }

    all_json_paths = [
        path for path in Path(root_dir).rglob(f"*{suffix}")
        if not any(substr in path.stem for substr in exclude_substrings)
    ]

    rows = []

    print(f"Found {len(all_json_paths)} JSON files to process")

    for path in tqdm(all_json_paths, desc="Computing composite difficulties"):
        try:
            with open(path, 'r') as f:
                metrics_dict = json.load(f)

            # Remove metadata keys
            metrics_dict.pop('label_mappings', None)
            metrics_dict.pop('errors', None)

            filename = path.stem
            dataset_id = filename.replace("Output_Multiclass__100_pct__", "").replace("_complexity_cuml", "")

            composite_df = compute_composite_difficulty_from_dict(
                metrics_dict, dataset_id, min_metrics_per_group
            )
            # display(composite_df)

            if composite_df is not None:
                rows.append(composite_df)

        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")

    if not rows:
        print("No datasets could be processed")
        return pd.DataFrame()

    result_df = pd.concat(rows, ignore_index=True)

    # Normalize across datasets
    print(f"\nNormalizing metrics across {len(result_df)} datasets...")

    # Determine metrics to normalize
    metric_cols = []
    for group_name, metric_list in metric_groups.items():
        metric_cols.extend(metric_list)

    available_metric_cols = [col for col in metric_cols if col in result_df.columns]

    # Fill NaNs before normalization if needed
    if result_df[available_metric_cols].isnull().values.any():
        print("[WARN] NaNs detected before normalization – filling with 0")
        result_df[available_metric_cols] = result_df[available_metric_cols].fillna(0)

    # Normalize
    if available_metric_cols:
        scaler = MinMaxScaler()
        result_df[available_metric_cols] = scaler.fit_transform(result_df[available_metric_cols])

    # Recompute group difficulties
    for group_name, metric_list in metric_groups.items():
        available_group_metrics = [m for m in metric_list if m in result_df.columns]
        if available_group_metrics:
            result_df[f"{group_name}_difficulty"] = result_df[available_group_metrics].mean(axis=1)

    # Recompute overall difficulty
    group_cols = [f"{group_name}_difficulty" for group_name in metric_groups.keys()
                  if f"{group_name}_difficulty" in result_df.columns]
    if group_cols:
        result_df["overall_difficulty"] = result_df[group_cols].mean(axis=1)

    print(f"Successfully processed {len(result_df)} datasets")
    print(f"Average metrics used per dataset: {result_df['metrics_used'].mean():.1f}")
    print(f"Average groups used per dataset: {result_df['groups_used'].mean():.1f}")

    return result_df


if __name__ == "__main__":

    ROOT_DIR = "../../2025-07-05/cuml_v2-20250705T065915Z-1-001"

    df_composite = compute_all_composite_difficulties(ROOT_DIR, min_metrics_per_group=1)

    df_composite.to_csv(os.path.join(ROOT_DIR), 'composite_metrics_raw.csv')
