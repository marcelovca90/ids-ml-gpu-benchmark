#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard library
import json
import os
import warnings
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Show all rows and prevent column truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)  # Auto-detect width


# In[2]:


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


# In[3]:


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


# In[4]:


row_order = [
    "CIC_IDS_2017_Multiclass",
    "CIC_IOT_Dataset2023_Multiclass",
    "IoT_23_Multiclass",
    "IoT_Network_Intrusion_Macro_Multiclass",
    "IoT_Network_Intrusion_Micro_Multiclass",
    "KDD_Cup_1999_Multiclass",
    "UNSW_NB15_Multiclass",
    "BCCC_CIC-BCCC-NRC-ACI-IOT-2023_Multiclass",
    "BCCC_CIC-BCCC-NRC-Edge-IIoTSet-2022_Multiclass",
    "BCCC_CIC-BCCC-NRC-IoMT-2024_Multiclass",
    "BCCC_CIC-BCCC-NRC-IoT-2022_Multiclass",
    "BCCC_CIC-BCCC-NRC-IoT-2023-Original_Training_and_Testing_Multiclass",
    "BCCC_CIC-BCCC-NRC-IoT-HCRL-2019_Multiclass",
    "BCCC_CIC-BCCC-NRC-MQTTIoT-IDS-2020_Multiclass",
    "BCCC_CIC-BCCC-NRC-TONIoT-2021_Multiclass",
    "BCCC_CIC-BCCC-NRC-UQ-IOT-2022_Multiclass",
    "BoT_IoT_Macro_Multiclass",
    "BoT_IoT_Micro_Multiclass",
    "CICAPT_IIoT_Phase1_Macro_Multiclass", # nok (single-class)
    "CICAPT_IIoT_Phase1_Micro_Multiclass", # nok (single-class)
    "CICAPT_IIoT_Phase2_Macro_Multiclass",
    "CICAPT_IIoT_Phase2_Micro_Multiclass",
    "CICEVSE2024_EVSE-A_Macro_Multiclass",
    "CICEVSE2024_EVSE-A_Micro_Multiclass",
    "CICEVSE2024_EVSE-B_Macro_Multiclass",
    "CICEVSE2024_EVSE-B_Micro_Multiclass",
    "CICIoMT2024_Bluetooth_Multiclass",
    "CICIoMT2024_WiFi_and_MQTT_Multiclass",
    "CICIoV2024_Decimal_Macro_Multiclass",
    "CICIoV2024_Decimal_Micro_Multiclass",
    "EDGE-IIOTSET_DNN-EdgeIIoT_Multiclass",
    "EDGE-IIOTSET_ML-EdgeIIoT_Multiclass",
    "MQTT_IoT_IDS2020_BiflowFeatures_Multiclass",
    "MQTT_IoT_IDS2020_PacketFeatures_Multiclass",
    "MQTT_IoT_IDS2020_UniflowFeatures_Multiclass",
    "NIDS_CIC-BoT-IoT_Multiclass",
    "NIDS_CIC-ToN-IoT_Multiclass",
    "NIDS_NF-BoT-IoT_Multiclass",
    "NIDS_NF-BoT-IoT-v2_Multiclass",
    "NIDS_NF-BoT-IoT-v3_Multiclass",
    "NIDS_NF-CICIDS2018-v3_Multiclass",
    "NIDS_NF-CSE-CIC-IDS2018_Multiclass",
    "NIDS_NF-CSE-CIC-IDS2018-v2_Multiclass",
    "NIDS_NF-ToN-IoT_Multiclass",
    "NIDS_NF-ToN-IoT-v2_Multiclass",
    "NIDS_NF-ToN-IoT-v3_Multiclass",
    "NIDS_NF-UNSW-NB15_Multiclass",
    "NIDS_NF-UNSW-NB15-v2_Multiclass",
    "NIDS_NF-UNSW-NB15-v3_Multiclass",
    "NIDS_NF-UQ-NIDS_Multiclass",
    "NIDS_NF-UQ-NIDS-v2_Multiclass",
    "N_BaIoT_Danmini_Doorbell_Multiclass",
    "N_BaIoT_Ecobee_Thermostat_Multiclass",
    "N_BaIoT_Ennio_Doorbell_Multiclass",
    "N_BaIoT_Philips_B120N10_Baby_Monitor_Multiclass",
    "N_BaIoT_Provision_PT_737E_Security_Camera_Multiclass",
    "N_BaIoT_Provision_PT_838_Security_Camera_Multiclass",
    "N_BaIoT_Samsung_SNH_1011_N_Webcam_Multiclass",
    "N_BaIoT_SimpleHome_XCS7_1002_WHT_Security_Camera_Multiclass",
    "N_BaIoT_SimpleHome_XCS7_1003_WHT_Security_Camera_Multiclass",
    "ToN_IoT_IoT_Fridge_Multiclass",
    "ToN_IoT_IoT_GPS_Tracker_Multiclass",
    "ToN_IoT_IoT_Garage_Door_Multiclass",
    "ToN_IoT_IoT_Modbus_Multiclass",
    "ToN_IoT_IoT_Motion_Light_Multiclass",
    "ToN_IoT_IoT_Thermostat_Multiclass",
    "ToN_IoT_IoT_Weather_Multiclass",
    "ToN_IoT_Linux_Disk_Multiclass",
    "ToN_IoT_Linux_Memory_Multiclass",
    "ToN_IoT_Linux_Process_Multiclass",
    "ToN_IoT_Network_Multiclass",
    "ToN_IoT_Windows_10_Multiclass",
    "ToN_IoT_Windows_7_Multiclass"
]


# In[5]:


def blend_with_white(rgb, alpha):
    return [1 - alpha * (1 - c) for c in rgb]

def format_and_color_columns(df, color_map_dict={}, alpha=0.0):
    df_colored = df.copy()

    for col in df.columns:
        col_data = df[col]

        # === Step 1: Apply your custom formatting ===
        if pd.api.types.is_float_dtype(col_data):
            if 'time' in col:
                formatted = col_data.map(lambda x: f"{x:,.1f}")
            elif 'size' in col:
                formatted = col_data.map(lambda x: f"{x:,.2f}")
            else:
                formatted = col_data.map(lambda x: f"{x:,.3f}")
        elif pd.api.types.is_integer_dtype(col_data):
            formatted = col_data.map(lambda x: f"{x:,}")
        else:
            formatted = col_data.astype(str)

        # === Step 2: Apply LaTeX color using colormap if specified ===
        if col in color_map_dict and pd.api.types.is_numeric_dtype(col_data):
            cmap = colormaps[color_map_dict[col]]
            valid_mask = col_data.notna()
            norm = Normalize(vmin=col_data[valid_mask].min(), vmax=col_data[valid_mask].max())
    
            # Start with string-typed formatted column
            colored_column = formatted.astype(str).copy()
    
            # Compute blended RGB
            rgba_colors = cmap(norm(col_data[valid_mask]))[:, :3]
            blended_colors = [blend_with_white(rgb, alpha=alpha) for rgb in rgba_colors]
    
            for i, (r, g, b) in zip(col_data[valid_mask].index, blended_colors):
                df_colored.loc[i, col] = (
                    f"\\cellcolor[rgb]{{{r:.3f}, {g:.3f}, {b:.3f}}} {formatted[i]}"
                )
        else:
            df_colored[col] = formatted

    return df_colored


# In[6]:


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

def compute_all_composite_difficulties(root_dir: str, suffix: str = ".complexity.json", min_metrics_per_group=1) -> pd.DataFrame:
    """
    Loads all complexity metric JSONs from a folder and computes composite difficulty scores.
    More flexible version that handles missing metrics gracefully.
    """
    
    exclude_substrings = {
        "CICAPT_IIoT_Phase1_Macro_Multiclass",
        "CICAPT_IIoT_Phase1_Micro_Multiclass",
        # "ToN_IoT_IoT_Motion_Light_Multiclass"
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
            # print(composite_df)
            
            if composite_df is not None:
                rows.append(composite_df)

        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")

    if not rows:
        print("No datasets could be processed")
        return pd.DataFrame()

    result_df = pd.concat(rows, ignore_index=True)
    # print(result_df)

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


# In[7]:


ROOT_DIR = "../../2025-11-17/Output_Zip_v4_Complexity"

df_composite = compute_all_composite_difficulties(ROOT_DIR, min_metrics_per_group=1)

print(df_composite)


# In[ ]:


# Set index to dataset_id
df_composite_indexed = df_composite.set_index('dataset_id')

# Filter row_order to only include datasets that exist in the DataFrame
valid_order = [d for d in row_order if d in df_composite_indexed.index]

# Reorder using the filtered list
df_ordered = df_composite_indexed.loc[valid_order]

# Export to LaTeX with thousands separator
df_ordered.style.format(thousands=",").to_latex("tables/complexity_metrics.tex")

df_ordered.to_excel('tables/complexity_metrics.xlsx')
df_ordered.to_json('tables/complexity_metrics.json', orient='index')


# In[ ]:


df_ordered


# In[ ]:


cols_to_drop = [
    "Feature_Relevance_difficulty", "Local_Overlap_difficulty", 
    "Boundary_Hardness_difficulty", "Global_Structure_difficulty", 
    "Class_Distribution_Separation_difficulty", 
    "metrics_used", "groups_used"
]

df_ordered_pretty = format_and_color_columns(
    df_ordered.drop(columns=cols_to_drop).loc[valid_order],
    color_map_dict={'overall_difficulty': 'RdYlGn_r'},
    alpha=0.5
)

with open("tables/table_2.tex", "w") as f:
    f.write(df_ordered_pretty.to_string())

df_ordered_pretty


# In[ ]:


# Compute min and max of overall_difficulty
vmin = df_ordered['overall_difficulty'].min()
vmax = df_ordered['overall_difficulty'].max()

# Select and style (don't include 'dataset_id' as it's now the index)
styled_df = df_ordered[['overall_difficulty', 'metrics_used', 'groups_used']].style \
    .background_gradient(subset=['overall_difficulty'], cmap='RdYlGn_r', vmin=vmin, vmax=vmax) \
    .format({'overall_difficulty': '{:.3f}'})

styled_df

