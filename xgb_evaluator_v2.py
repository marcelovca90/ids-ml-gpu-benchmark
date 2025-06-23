#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -r requirements.txt


# In[2]:


dataset_path = '2025-06-13/Multiclass/NIDS_NF-UQ-NIDS-v2_Multiclass.parquet'
target_column = 'label'
sampling_rate_global = None # 0.10
sampling_rate_sets = 0.10
sample_sets = ['train']
min_samples_per_class = 1
feature_selection_threshold = 0.99
sample_filtering_quantile = 0.10
hpo_n_trials = 10 # 1000
hpo_timeout = 60 # 3600
num_boost_round = 100 # 500
early_stopping_rounds = 10 # 50
n_jobs = -1
random_state = 42
plot_param_importances = False


# In[3]:


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# In[4]:


import os

os.makedirs('deploy/multiclass', exist_ok=True)


# # Step 1: Data Preprocessing

# In[5]:


import cudf
import cupy as cp
import numpy as np
import random
import time

random.seed(42)
cp.random.seed(42)
np.random.seed(42)

global_start = time.time()

df_full = cudf.read_parquet(dataset_path)

df_full = df_full[df_full[target_column] != 'mqtt_bruteforce'].copy()

if sampling_rate_global:
    df_full = df_full.sample(frac=sampling_rate_global, random_state=random_state)

print(df_full.head(10))

print(df_full[target_column].value_counts())


# In[6]:


import json

with open(dataset_path.replace('.parquet', '.json'), 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Step 2: Apply dtypes from metadata to df
for col, dtype in metadata["dtypes"].items():
    df_full[col] = df_full[col].astype(dtype)


# In[8]:


# Factorize target column
df_full[target_column], unique_values = df_full[target_column].factorize()
assert df_full[target_column].nunique() > 2, "Multiclass classification requires more than 2 classes."
label_to_index = {value: i for i, value in enumerate(unique_values.to_pandas())}
index_to_label = {v: k for k, v in label_to_index.items()}
labels = list(label_to_index.keys())
df_full[target_column] = df_full[target_column].astype('int8')

# Detect and assign codes to categorical columns
numeric_columns = df_full.drop(columns=[target_column]).select_dtypes(include=['number']).columns.tolist()
categorical_cols = df_full.drop(columns=[target_column]).select_dtypes(include=['category']).columns.tolist()

df_full[target_column].value_counts()


# In[12]:


from cuml.model_selection import train_test_split

def ensure_min_samples_per_class(df, stratify_col, min_samples_per_class, random_state):
    """Ensures that each class has at least `min_samples_per_class` samples using oversampling."""
    class_counts = df[stratify_col].value_counts().to_pandas()

    # Oversample minority classes
    oversampled = [
        df[df[stratify_col] == c].sample(n=min_samples_per_class, replace=True, random_state=random_state)
        for c in class_counts[class_counts < min_samples_per_class].index
    ]
    df_oversampled = cudf.concat([df] + oversampled, ignore_index=True) if oversampled else df

    return df_oversampled.reset_index(drop=True)  # Fix index mismatch

def restore_dtypes(df, dtypes):
    for col, dtype in dtypes.to_dict().items():
        df[col] = df[col].astype(dtype)

def assign_subsets(df, stratify_col, train_frac, val_frac, test_frac, random_state):
    """Splits data into mutually exclusive train/val/test subsets before applying stratified sampling."""
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1"

    # Assign subset labels
    df_train, df_temp = train_test_split(df, test_size=(1 - train_frac), stratify=df[stratify_col], random_state=random_state)
    df_val, df_test = train_test_split(df_temp, test_size=(test_frac / (val_frac + test_frac)), stratify=df_temp[stratify_col], random_state=random_state)

    df_train["subset"] = "train"
    df_val["subset"] = "val"
    df_test["subset"] = "test"

    return cudf.concat([df_train, df_val, df_test]).reset_index(drop=True)

def sample_group(x):
    """Helper function to stratify sample while ensuring class presence."""
    n_samples = max(min_samples_per_class, int(len(x) * sampling_rate_sets))
    return x.sample(n=n_samples, replace=len(x) < n_samples, random_state=random_state)

def stratified_sample(df, stratify_col, sample_sets, sampling_rate_sets, min_samples_per_class, random_state):
    """Applies stratified sampling while ensuring minimum samples per class, only for selected subsets."""

    # Apply stratified sampling only for the requested subsets
    df_sampled = df.groupby(["subset", stratify_col], group_keys=False).apply(
        lambda x: sample_group(x) if x["subset"].iloc[0] in sample_sets else x
    )

    return df_sampled.reset_index(drop=True)


# In[13]:

df_dtypes_before = df_full.dtypes.copy(deep=True).sort_index()

# Ensure enough samples before splitting
split_ok = False
while not split_ok:
    df_full = ensure_min_samples_per_class(df_full, target_column, min_samples_per_class, random_state)
    df_full = df_full.reset_index(drop=True)  # Reset index before splitting
    df_dtypes_backup = df_full.dtypes.copy(deep=True)

    # Factorize for stratification
    category_mappings = {}

    for col, dtype in df_full.dtypes.to_dict().items():
        if dtype == 'category':
            codes, uniques = df_full[col].factorize()
            df_full[col] = codes.astype('int32')
            category_mappings[col] = uniques.to_pandas().tolist()

    try:
        # Assign mutually exclusive train/val/test subsets
        df_full = assign_subsets(df_full, "label", train_frac=0.6, val_frac=0.2, test_frac=0.2, random_state=random_state)
        split_ok = True  # If it succeeds, exit loop
    except ValueError as e:
        print(f"Resampling due to insufficient class representation (min_samples_per_class={min_samples_per_class})...")
        min_samples_per_class += 1  # Increment dynamically and retry
    
    for col, dtype in df_dtypes_backup.items():
        if col in category_mappings:
            # Restore category values from factorized codes
            mapping = dict(enumerate(category_mappings[col]))
            df_full[col] = df_full[col].map(mapping).astype('category')
        else:
            # Restore other dtypes (numeric, bool, etc.)
            df_full[col] = df_full[col].astype(dtype)

print(f"Minimal oversampling completed successfully (min_samples_per_class={min_samples_per_class}).")

df_full[target_column].value_counts()

df_dtypes_after = df_full.dtypes.copy(deep=True).drop('subset').sort_index()

assert df_dtypes_before.equals(df_dtypes_after), "Dtype mismatch after assigning subsets"

# In[14]:


# Fill missing categorical values
def fill_categorical_nas(dfs):
    return
    for df in dfs:
        for col in df.select_dtypes(include=['category']).columns:
            if df[col].to_pandas().isna().sum() > 0:
                df[col] = cudf.Series(df[col].to_pandas().astype(str).replace("nan", "missing").astype("category"))

# Extract final datasets
df_train_full = df_full[df_full["subset"] == "train"].drop(columns=["subset"])
df_val_full = df_full[df_full["subset"] == "val"].drop(columns=["subset"])
df_test_full = df_full[df_full["subset"] == "test"].drop(columns=["subset"])

# Convert back to X, y format
X_train_full, y_train_full = df_train_full.drop(columns=[target_column]), df_train_full[target_column]
X_val_full, y_val_full = df_val_full.drop(columns=[target_column]), df_val_full[target_column]
X_test_full, y_test_full = df_test_full.drop(columns=[target_column]), df_test_full[target_column]

fill_categorical_nas([X_train_full, X_val_full, X_test_full])

# Ensure disjoint splits
assert set(X_train_full.to_pandas().index).isdisjoint(set(X_val_full.to_pandas().index)), "Train and Validation sets are not disjoint!"
assert set(X_train_full.to_pandas().index).isdisjoint(set(X_test_full.to_pandas().index)), "Train and Test sets are not disjoint!"
assert set(X_val_full.to_pandas().index).isdisjoint(set(X_test_full.to_pandas().index)), "Validation and Test sets are not disjoint!"

# Ensure all classes are present in each split
assert y_train_full.nunique() == df_full[target_column].nunique(), "Some classes are missing in Train!"
assert y_val_full.nunique() == df_full[target_column].nunique(), "Some classes are missing in Validation!"
assert y_test_full.nunique() == df_full[target_column].nunique(), "Some classes are missing in Test!"

# Print final distributions
print(f"Train      : {len(df_train_full)} samples ({(100.0 * len(df_train_full) / len(df_full)):.2f}%), {y_train_full.nunique()} unique classes ({sorted(y_train_full.to_pandas().unique().tolist())})")
print(f"Validation : {len(df_val_full)} samples ({(100.0 * len(df_val_full) / len(df_full)):.2f}%), {y_val_full.nunique()} unique classes ({sorted(y_train_full.to_pandas().unique().tolist())})")
print(f"Test       : {len(df_test_full)} samples ({(100.0 * len(df_test_full) / len(df_full)):.2f}%), {y_test_full.nunique()} unique classes ({sorted(y_train_full.to_pandas().unique().tolist())})")


# # Utils

# In[15]:


from sklearn.metrics import f1_score
import numpy as np
import tempfile
import xgboost as xgb

num_classes = df_full[target_column].nunique()
loss_function = 'binary:logistic' if num_classes == 2 else 'multi:softprob'
eval_metric = 'logloss' if num_classes == 2 else 'mlogloss'
n_folds, shuffle, stratify = 5, True, True

def get_model_size(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ubj") as temp_model:
        model_path = temp_model.name
    model.save_model(model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    os.remove(model_path)
    return round(model_size_mb, 2)

# Custom F1-score evaluation function
def f1_weighted_eval(preds, dtrain):
    y_true = dtrain.get_label()
    # Convert probabilities to class labels
    if num_classes > 2:
        y_pred = np.argmax(preds.reshape(y_true.shape[0], -1), axis=1)
    else:
        y_pred = (preds > 0.5).astype(int)
    # Compute weighted F1-score
    f1 = f1_score(y_true, y_pred, average="weighted")
    return "f1_weighted", f1

# Default XGB Booster parameters
default_booster_params = {
    "objective": loss_function,                     # Multi-class classification
    "early_stopping_rounds": early_stopping_rounds, # Number of iterations
    "num_boost_round": num_boost_round,             # Number of CV folds
    "eval_metric": eval_metric,                     # Log loss
    "device": 'cuda'                                # GPU (CUDA)
}
if num_classes > 2:
    default_booster_params["num_class"] = num_classes # Only for multi-class classification

# Default XGB CV parameters
default_cv_params = {
    "params": default_booster_params,               # Booster parameters
    "early_stopping_rounds": early_stopping_rounds, # Number of rounds before early stopping
    "num_boost_round": num_boost_round,             # Number of iterations
    "nfold": n_folds,                               # Number of CV folds
    "shuffle": shuffle,                             # Shuffle samples before creating folds
    "stratified": stratify,                         # Stratify classes on CV split
    "metrics": eval_metric,                         # Log loss
    "feval": f1_weighted_eval,                      # Monitor weighted F1 score
    "seed": random_state                            # Passed to numpy.random.seed
}

# Default XGB training parameters
default_train_params = {
    "early_stopping_rounds": early_stopping_rounds, # Number of rounds before early stopping
    "num_boost_round": num_boost_round,             # Number of iterations
}

# Function to Train, Validate, and Test XGBoost Model
def train_xgb(
    X_train, X_val, X_test, y_train, y_val, y_test,
    cv=False,
    custom_booster_params=None,
    persist: bool = True,
    filename: str = "deploy/multiclass/xgb_model.json",
    verbose=1
):
    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True, feature_names=X_val.columns.tolist()) if X_val is not None else None
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True, feature_names=X_test.columns.tolist()) if X_test is not None else None

    # Merge default booster params with custom ones
    booster_params = {**default_booster_params, **(custom_booster_params or {})}

    if cv:
        cv_params = {
            **default_cv_params,
            "params": booster_params,
            "verbose_eval": verbose
        }
        training_start = time.time()
        results = xgb.cv(dtrain=dtrain, **cv_params)
        training_end = time.time()
        training_time = (training_end - training_start) / n_folds
        latency = training_time / len(y_train)
        f1_score_ans = float(results['test-f1_weighted-mean'].mean())
        model, model_size = None, None
    else:
        training_start = time.time()
        model = xgb.train(
            params=booster_params,
            dtrain=dtrain,
            evals=[(dtrain, "Train"), (dval, "Validation")],
            **default_train_params
        )
        training_end = time.time()

        # Predict
        y_pred_probs = model.predict(dtest)
        y_pred = np.argmax(y_pred_probs, axis=1) if num_classes > 2 else (y_pred_probs > 0.5).astype(int)

        latency = (time.time() - training_end) / len(y_test)
        f1_score_ans = float(f1_score(y_test.to_numpy(), y_pred, average="weighted"))
        training_time = training_end - training_start
        model_size = get_model_size(model)

        # Persist the model properly
        if persist:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            model.save_model(filename)  # Save as .json or .bst

    return model, training_time, latency, f1_score_ans, model_size


# In[16]:


# train_xgb(X_train_full, None, None, y_train_full, None, None, cv=True)


# In[17]:


# train_xgb(X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full, cv=False)


# # Step 1: Baseline Evaluation (full dataset)

# In[18]:


# Train and evaluate using train_xgb function
model_full, train_time_full, latency_full, f1_weighted_full, model_size_full = train_xgb(
    X_train_full, X_val_full, X_test_full,
    y_train_full, y_val_full, y_test_full,
    cv=False,  # Cross-validation only used during HPO
    persist=True,
    filename="deploy/multiclass/xgb_full.json"
)

# Print results
print(f"Training Time (full): {train_time_full:.3f} seconds")
print(f"Latency (full): {latency_full:.2e} seconds")
print(f"Weighted F1-Score (full): {f1_weighted_full:.6f}")
print(f"Model Size: {model_size_full} MB")


# # Step 2.1: Sampling

# In[19]:


# Apply stratified sampling only to the selected subsets
df_sampled = stratified_sample(df_full, target_column, sample_sets, sampling_rate_sets, min_samples_per_class, random_state)

# Extract final datasets
df_train_sampled = df_sampled[df_sampled["subset"] == "train"].drop(columns=["subset"])
df_val_sampled = df_sampled[df_sampled["subset"] == "val"].drop(columns=["subset"])
df_test_sampled = df_sampled[df_sampled["subset"] == "test"].drop(columns=["subset"])

# Convert back to X, y format
X_train_sampled, y_train_sampled = df_train_sampled.drop(columns=[target_column]), df_train_sampled[target_column]
X_val_sampled, y_val_sampled = df_val_sampled.drop(columns=[target_column]), df_val_sampled[target_column]
X_test_sampled, y_test_sampled = df_test_sampled.drop(columns=[target_column]), df_test_sampled[target_column]

fill_categorical_nas([X_train_sampled, X_val_sampled, X_test_sampled])

# Ensure disjoint splits
assert set(X_train_sampled.to_pandas().index).isdisjoint(set(X_val_sampled.to_pandas().index)), "Train and Validation sets are not disjoint!"
assert set(X_train_sampled.to_pandas().index).isdisjoint(set(X_test_sampled.to_pandas().index)), "Train and Test sets are not disjoint!"
assert set(X_val_sampled.to_pandas().index).isdisjoint(set(X_test_sampled.to_pandas().index)), "Validation and Test sets are not disjoint!"

# Ensure all classes are present in each split
assert y_train_sampled.nunique() == df_sampled[target_column].nunique(), "Some classes are missing in Train!"
assert y_val_sampled.nunique() == df_sampled[target_column].nunique(), "Some classes are missing in Validation!"
assert y_test_sampled.nunique() == df_sampled[target_column].nunique(), "Some classes are missing in Test!"

# Print final distributions
print(f"Train      : {len(df_train_sampled)} samples ({(100.0 * len(df_train_sampled) / len(df_full)):.2f}%), {y_train_sampled.nunique()} unique classes ({sorted(y_train_sampled.to_pandas().unique().tolist())})")
print(f"Validation : {len(df_val_sampled)} samples ({(100.0 * len(df_val_sampled) / len(df_full)):.2f}%), {y_val_sampled.nunique()} unique classes ({sorted(y_train_sampled.to_pandas().unique().tolist())})")
print(f"Test       : {len(df_test_sampled)} samples ({(100.0 * len(df_test_sampled) / len(df_full)):.2f}%), {y_test_sampled.nunique()} unique classes ({sorted(y_train_sampled.to_pandas().unique().tolist())})")


# # Step 2.2: Updated Evaluation (with Sampling)

# In[20]:


# Train and evaluate using train_xgb function
model_sampled, train_time_sampled, latency_sampled, f1_weighted_sampled, model_size_sampled = train_xgb(
    X_train_sampled, X_val_sampled, X_test_sampled,
    y_train_sampled, y_val_sampled, y_test_sampled,
    cv=False,  # Cross-validation only used during HPO,
    persist=True,
    filename="deploy/multiclass/xgb_sampled.json"
)

# Print results
print(f"Training Time (sampled): {train_time_sampled:.3f} seconds")
print(f"Latency (sampled): {latency_sampled:.2e} seconds")
print(f"Weighted F1-Score (sampled): {f1_weighted_sampled:.6f}")
print(f"Model Size: {model_size_sampled} MB")


# # Step 3.1: Feature Selection

# In[21]:


import matplotlib.pyplot as plt

# Calculate and sort features by importance
feature_importance = model_sampled.get_score(importance_type="gain")
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
features, importance = zip(*sorted_features)

# Create the plot
plt.figure(figsize=(10, 8))
bars = plt.barh(features, importance, color='skyblue')

# Annotate each bar with its importance value
for bar, value in zip(bars, importance):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f" {value:.3f}", va='center')

# Labels and title
plt.xlabel("Gain (Importance)")
plt.ylabel("Features")
plt.title("Feature Importance (Gain)")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid for better readability

# plt.show()


# In[22]:


# Get feature importance
feature_importance = model_sampled.get_score(importance_type="gain")
importance_df = cudf.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Normalize importance scores
importance_df["Cumulative_Importance"] = importance_df["Importance"].cumsum() / importance_df["Importance"].sum()

# Find the smallest set of features that explains at least 95% of the importance
N = np.argmax(importance_df["Cumulative_Importance"] >= feature_selection_threshold) + 1  # Adjust threshold as needed

# Select top features
selected_features = importance_df["Feature"][:N].to_pandas().tolist()

print(f"Optimal number of features: {N} (from {len(feature_importance)})")
importance_df


# # Step 3.2: Updated Evaluation (with Sampling and Feature Selection)

# In[23]:


# Reduce the feature space for train, validation, and test sets
X_train_reduced = X_train_sampled[selected_features]
X_val_reduced = X_val_sampled[selected_features]
X_test_reduced = X_test_sampled[selected_features]

fill_categorical_nas([X_train_reduced, X_val_reduced, X_test_reduced])

# Train and evaluate using train_xgb function
model_reduced, train_time_reduced, latency_reduced, f1_weighted_reduced, model_size_reduced = train_xgb(
    X_train_reduced, X_val_reduced, X_test_reduced,
    y_train_sampled, y_val_sampled, y_test_sampled,
    cv=False,  # Cross-validation only used during HPO
    persist=True,
    filename="deploy/multiclass/xgb_reduced.json"
)

# Print results
print(f"Training Time (reduced): {train_time_reduced:.3f} seconds")
print(f"Latency (reduced): {latency_reduced:.2e} seconds")
print(f"Weighted F1-Score (reduced): {f1_weighted_reduced:.6f}")
print(f"Model Size (reduced): {model_size_reduced} MB")


# # Step 4.1: Row Filtering

# In[24]:


def convert_categorical_to_frequency(df, normalize=True):
    """Convert categorical features to frequency encoding with reversibility support."""
    df_encoded = df.copy()
    category_mappings = {}
    row_mappings = {}

    for col in df_encoded.select_dtypes(include=['category']).columns:
        # Calculate normalized frequencies
        freqs = df[col].to_pandas().value_counts().astype('float32')
        if normalize:
            freqs = freqs / len(df)
        freq_map = freqs.to_dict()

        # Encode column
        df_encoded[col] = df[col].to_pandas().map(freq_map).astype('float32')

        # Save both mappings
        category_mappings[col] = freq_map
        row_mappings[col] = df[col].reset_index(drop=True)

    return df_encoded, category_mappings, row_mappings


def revert_frequency_encoding(df_encoded, row_mappings):
    """Revert frequency-encoded DataFrame to original categories using stored row-wise values."""
    df_reverted = df_encoded.copy().reset_index(drop=True)

    for col in row_mappings:
        if col in df_reverted.columns:
            df_reverted[col] = row_mappings[col]  # restore from stored original

    return df_reverted


def filter_low_mean_samples(X_encoded: cudf.DataFrame, y: cudf.Series, quantile_threshold: float):
    """Filter low-mean samples per class while preserving all class labels."""
    
    # Step 1: Combine features and label
    combined_df = X_encoded.copy()
    combined_df['target'] = y

    # Step 2: Compute row means once
    row_means = X_encoded.mean(axis=1)
    combined_df['row_mean'] = row_means

    # Step 3: Compute quantile thresholds per class
    class_thresholds = (
        combined_df[['target', 'row_mean']]
        .groupby('target')
        .quantile(q=quantile_threshold)
        .rename(columns={'row_mean': 'threshold'})
        .reset_index()
    )

    # Step 4: Join thresholds back to combined_df
    combined_df = combined_df.merge(class_thresholds, on='target', how='left')

    # Step 5: Apply filtering
    filtered_df = combined_df[combined_df['row_mean'] > combined_df['threshold']]

    # Step 6: Ensure all classes are represented
    present_classes = set(cp.asnumpy(filtered_df['target'].unique()))
    all_classes = set(cp.asnumpy(y.unique()))
    missing_classes = all_classes - present_classes

    for label in missing_classes:
        class_df = combined_df[combined_df['target'] == label]
        best_idx = class_df['row_mean'].to_pandas().idxmax()
        best_sample = class_df.loc[[best_idx]]
        filtered_df = cudf.concat([filtered_df, best_sample])

    # Step 7: Final separation
    X_filtered = filtered_df.drop(['target', 'row_mean', 'threshold'], axis=1)
    y_filtered = filtered_df['target']

    return X_filtered, y_filtered


# In[25]:


# Apply frequency encoding to the training set
print('convert_categorical_to_frequency')
X_train_filtered_encoded, cat_map, row_map = convert_categorical_to_frequency(X_train_reduced)

# Apply per-class row filtering
print('filter_low_mean_samples')
X_train_filtered, y_train_filtered = filter_low_mean_samples(
    X_train_filtered_encoded, 
    y_train_sampled, 
    sample_filtering_quantile
)

assert set(X_train_filtered.columns) == set(X_train_reduced.columns)

# Revert frequency encoding
print('revert_frequency_encoding')
# Subset row_map to match the filtered row indices
filtered_row_map = {col: row_map[col].loc[X_train_filtered.index] for col in row_map}
X_train_filtered = revert_frequency_encoding(X_train_filtered, row_map)
fill_categorical_nas([X_train_filtered])

assert set(X_train_filtered.columns) == set(X_train_reduced.columns)

# Print results
print(f"Original training set size: {len(X_train_reduced)} rows")
print(f"Filtered training set size: {len(X_train_filtered)} rows")

print(f"\nClass distribution before filtering:")
print(y_train_sampled.value_counts().sort_index())
print(f"\nClass distribution after filtering:")
print(y_train_filtered.value_counts().sort_index())

# Final validations
assert len(X_train_filtered) == len(y_train_filtered), "Sample count mistmatch between X and y"
assert len(X_train_filtered.columns) == len(X_train_reduced.columns), "Column count changed"
assert set(X_train_filtered.columns) == set(X_train_reduced.columns), "Column names changed"
assert y_train_filtered.nunique() == y_train_sampled.nunique(), "Class count changed"
assert y_train_filtered.nunique() == df_full[target_column].nunique(), "Some classes are missing"
assert all(y_train_filtered.to_pandas().value_counts() >= 1), "Some classes have zero samples"
assert X_train_filtered.index.is_unique, "Duplicate indices in filtered data"

# # Step 4.2: Updated Evaluation (with Sampling, Feature Selection, and Row Filtering)

# In[26]:


# Train and evaluate using train_xgb function
clf_filtered, train_time_filtered, latency_filtered, f1_weighted_filtered, model_size_filtered = train_xgb(
    X_train_filtered, X_val_reduced, X_test_reduced,
    y_train_filtered, y_val_sampled, y_test_sampled,
    cv=False,  # Cross-validation only used during HPO
    persist=True,
    filename="deploy/multiclass/xgb_filtered.json"
)

# Print results
print(f"Training Time (filtered): {train_time_filtered:.3f} seconds")
print(f"Latency (filtered): {latency_filtered:.2e} seconds")
print(f"Weighted F1-Score (filtered): {f1_weighted_filtered:.6f}")
print(f"Model Size (filtered): {model_size_filtered} MB")


# # Step 5.1: HPO (Preprocessing Wrappers)

# ### Numeric scalers

# In[27]:


scaling_methods = ['none', 'maxabs', 'minmax', 'norm', 'robust', 'standard'] if numeric_columns else ['none']


# In[28]:


from cuml.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler

def make_numeric_scaler(scaling_method, X):
    numeric_columns = X.select_dtypes(include=['number']).columns
    non_numeric_columns = X.columns.difference(numeric_columns)

    scalers = {
        'none': None,
        'maxabs': MaxAbsScaler(),
        'minmax': MinMaxScaler(),
        'norm': Normalizer(),
        'robust': RobustScaler(),
        'standard': StandardScaler()
    }

    scaler = scalers.get(scaling_method)

    def scale_numeric_features(X_input, fit=False):
        X_numeric_pd = X_input[numeric_columns].copy()
        X_non_numeric_pd = X_input[non_numeric_columns].copy()
        X_output = None

        if scaling_method == 'none':
            X_output = X_input
        else:
            if fit:
                scaler.fit(X_numeric_pd)
            X_numeric_scaled_pd = scaler.transform(X_numeric_pd)
            X_numeric_scaled_cu = X_numeric_scaled_pd.astype('float32') # cudf.from_pandas(X_numeric_scaled_pd)
            X_numeric_scaled_cu.columns = list(numeric_columns)
            X_numeric_scaled_cu.index = X_input.index
            # X_numeric_scaled_cu = X_numeric_scaled_cu.reset_index(drop=True)
            X_non_numeric_cu = X_non_numeric_pd # cudf.from_pandas(X_non_numeric_pd)
            X_non_numeric_cu.columns = list(non_numeric_columns)
            X_non_numeric_cu.index = X_input.index
            # X_non_numeric_cu = X_non_numeric_cu.reset_index(drop=True)
            X_output = cudf.concat([X_numeric_scaled_cu, X_non_numeric_cu], axis=1)
            assert X_input.shape == X_output.shape, "Shape mismatch after concat."

        return X_output

    return scale_numeric_features


# ### Categoric encoders

# In[29]:


encoding_methods = ['none', 'onehot', 'ordinal', 'frequency'] if categorical_cols else ['none']
# encoding_methods = ['frequency'] if categorical_cols else ['none']


# In[30]:


from cuml.preprocessing import LabelEncoder as CumlLabelEncoder
from cuml.preprocessing import OneHotEncoder as CumlOneHotEncoder


# Define categorical encoding function
def make_categorical_encoder(encoding_method, X, max_categories_for_onehot=10):
    """Creates an encoder for categorical features while keeping numerical columns unchanged."""

    categorical_columns = X.select_dtypes(include=['category']).columns
    numeric_columns = X.columns.difference(categorical_columns)  # Numeric columns remain unchanged

    # Split categorical columns for One-Hot vs. None Encoding
    onehot_columns = [col for col in categorical_columns if X[col].nunique() <= max_categories_for_onehot]
    none_columns = list(set(categorical_columns) - set(onehot_columns))  # High-cardinality features

    # Ensure onehot_columns is a valid list (avoiding Index issues)
    onehot_columns = list(onehot_columns) if hasattr(onehot_columns, 'tolist') else onehot_columns

    # Define encoding strategies
    encoders = {
        'none': None,   # No transformation (Handled separately)
        'onehot': CumlOneHotEncoder(handle_unknown='ignore', sparse=False, sparse_output=False),
        'ordinal': CumlLabelEncoder(handle_unknown='ignore'),
        'frequency': None  # Frequency encoding requires custom logic
    }

    encoder = encoders.get(encoding_method)

    # Store frequency map for reuse
    frequency_col_map, frequency_row_map = {}, {}

    def encode_categorical_features(X_input, fit=False):
        nonlocal encoder
        """Encodes categorical features while keeping numeric features unchanged."""
        _categorical_columns = X_input.select_dtypes(include=['category']).columns
        _numeric_columns = X_input.columns.difference(_categorical_columns)
        X_numeric = X_input[_numeric_columns].copy()
        X_categorical = X_input[_categorical_columns].copy()

        if encoding_method == 'none':
            # Convert categorical columns to category dtype and assign codes
            # X_categorical = X_categorical.apply(lambda col: col.astype('category').cat.codes)
            return X_input

        elif encoding_method == 'onehot':
            X_numeric_copy = X_numeric.copy()  # Start with numeric columns
            # Handle categorical columns together for cuML
            if fit:
                encoder.fit(X_categorical[onehot_columns])
            X_onehot = cudf.DataFrame(
                data=encoder.transform(X_categorical[onehot_columns]).astype('int8'), 
                columns=encoder.get_feature_names(onehot_columns)
            )
            # Ensure index alignment
            X_onehot.index = X_numeric_copy.index  
            # Add encoded columns to result
            X_encoded = cudf.concat([X_numeric_copy, X_onehot], axis=1)

            # Apply 'none' encoding (category codes) for high-cardinality features
            if none_columns:
                # X_none_encoded = X_categorical[none_columns].apply(lambda col: col.cat.codes)
                X_encoded = cudf.concat([X_encoded, X_input[none_columns]], axis=1)

            return X_encoded

        elif encoding_method == 'ordinal':
            X_numeric_copy = X_numeric.copy()  # Start with numeric columns
            X_ordinal_partials = []
            # Handle each categorical column separately for cuML
            for col in X_categorical.columns:
                if fit:
                    encoder.fit(X_categorical[col])
                X_ordinal = cudf.DataFrame(
                    data=encoder.transform(X_categorical[col]).astype('category'),
                    columns=[col]
                )
                # Ensure index alignment
                X_ordinal.index = X_numeric_copy.index  
                # Add encoded columns to result
                X_ordinal_partials.append(X_ordinal)
            X_encoded = cudf.concat([X_numeric_copy, cudf.concat(X_ordinal_partials, axis=1)], axis=1)

            return X_encoded

        elif encoding_method == 'frequency':
            nonlocal frequency_col_map, frequency_row_map
            X_numeric_copy = X_numeric.copy()  # Start with numeric columns

            if fit:
                # Fit and encode
                X_categorical_encoded, frequency_col_map, frequency_row_map = \
                    convert_categorical_to_frequency(X_categorical)
            else:
                # Transform using saved frequency map
                X_categorical_encoded = X_categorical.copy()
                for col in X_categorical.columns:
                    mapped = X_categorical[col].to_pandas().map(frequency_col_map.get(col, {}))
                    X_categorical_encoded[col] = cudf.Series(mapped, index=X_categorical.index).astype('float32')

            return cudf.concat([X_numeric_copy, X_categorical_encoded], axis=1)

        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")

        # Ensure all categorical columns are replaced by their codes
        # X_categorical = X_categorical.apply(lambda col: col.cat.codes)

    return encode_categorical_features


# # Step 5.2: HPO (Data Balancing wrappers)

# In[31]:


from cuml.neighbors import NearestNeighbors as CumlNearestNeighbors


# In[32]:


def value_counts_to_dict(array):
    unique, counts = np.unique(array, return_counts=True)
    value_counts_dict = dict(zip(unique, counts))
    return value_counts_dict


# In[33]:


def fit_resample(_X_train, _y_train, over_method, over_thresh, under_method, under_thresh):

    if over_method == "smotenc" and under_method == "none":
        pass
    _X_names = _X_train.columns.tolist()
    _y_name = _y_train.name

    _X_train_copy = _X_train.copy()
    _y_train_copy = _y_train.copy()

    if over_thresh:
        value_counts = _y_train_copy.value_counts().to_dict()
        n_neighbors = min(5, min(value_counts.values()))
        n_generate = build_oversampling_strategy(value_counts, over_thresh)
        over_strategy = patch_oversampling_strategy(value_counts, n_generate)
        cat_features = [
            _X_train_copy.columns.get_loc(col)
            for col in _X_train_copy.select_dtypes(include=['category']).columns
        ]
        over_sampler = make_over_sampler(over_method, over_strategy, n_neighbors, cat_features)
        _X_train_copy, _y_train_copy = over_sampler.fit_resample(_X_train_copy, _y_train_copy)

    if under_thresh:
        value_counts = _y_train_copy.value_counts().to_dict()
        n_remove = build_undersampling_strategy(value_counts, under_thresh)
        under_strategy = patch_undersampling_strategy(value_counts, n_remove)
        under_sampler = make_under_sampler(under_method, under_strategy)
        _X_train_copy, _y_train_copy = under_sampler.fit_resample(_X_train_copy, _y_train_copy)

    return cudf.DataFrame(_X_train_copy, columns=_X_names), cudf.Series(_y_train_copy, name=_y_name)



# ### Oversampling

# In[34]:


over_method_choices = ['random']#, 'smotenc']


# In[35]:


over_threshold_choices = [float(f) for f in np.linspace(0, 4, num=17).round(2)] + ['auto']


# In[36]:

from cudf_resampler import CumlRandomResampler
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler

def build_oversampling_strategy(value_counts, threshold):
    n_occurrences = sum(value_counts.values())
    perfectly_balanced_occurrences = int(n_occurrences / len(value_counts.keys()))

    if threshold == "auto":
        n_generate = {
            class_: perfectly_balanced_occurrences - occ
                    if occ < perfectly_balanced_occurrences else 0
                    for class_, occ in value_counts.items()
        }
    else:
        n_generate = {
            class_: int(min(occ * threshold, perfectly_balanced_occurrences - occ))
                    if occ < perfectly_balanced_occurrences else 0
                    for class_, occ in value_counts.items()
        }
    return n_generate

def patch_oversampling_strategy(value_counts, n_generate):
    return {k: (value_counts[k] + n_generate[k]) for k in value_counts.keys()}

def make_over_sampler(over_method, over_strategy, n_neighbors, cat_features=None):
    if over_method == "random":
        return CumlRandomResampler(strategy=over_strategy, sampling_type="over")
    elif over_method == "smote":
        return SMOTE(k_neighbors=CumlNearestNeighbors(n_neighbors=n_neighbors), sampling_strategy=over_strategy)
    elif over_method == "smotenc":
        return SMOTENC(categorical_features=cat_features, k_neighbors=CumlNearestNeighbors(n_neighbors=n_neighbors), sampling_strategy=over_strategy)
    else:
        raise ValueError(f"Unknown oversampling method: {over_method}")


# ### Undersampling

# In[37]:


under_method_choices = ['random']#, 'tomek']


# In[38]:


under_threshold_choices = [float(f) for f in np.linspace(0, 0.95, num=20).round(2)] + ['auto']


# In[39]:


from imblearn.under_sampling import TomekLinks as TomekLinksImblearn
from sklearn.utils import _safe_indexing  # Needed for compatibility

class TomekLinksCUDA(TomekLinksImblearn):
    def fit_resample(self, X, y):
        nn = CumlNearestNeighbors(n_neighbors=2)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1]

        links = self.is_tomek(y, nns, self.sampling_strategy_)
        self.sample_indices_ = np.flatnonzero(np.logical_not(links))

        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_),
        )


# In[40]:


from imblearn.under_sampling import RandomUnderSampler

def build_undersampling_strategy(value_counts, threshold):
    n_occurences = sum([n for n in value_counts.values()])
    perfectly_balanced_occurences = int(n_occurences / len(value_counts.keys()))
    if threshold == "auto":
        n_remove = {
            class_: occ - perfectly_balanced_occurences
                    if occ > perfectly_balanced_occurences else 0
                    for class_, occ in value_counts.items()
        }
    else:
        n_remove = {
            class_: int(min(occ * threshold, occ - perfectly_balanced_occurences))
            if occ > perfectly_balanced_occurences else 0
            for class_, occ in value_counts.items()
        }
    return n_remove

def patch_undersampling_strategy(value_counts, n_remove):
    return {k : (value_counts[k] - n_remove[k]) for k in value_counts.keys()}

def make_under_sampler(under_method, under_strategy):
    if under_method == "random":
        return CumlRandomResampler(strategy=under_strategy, sampling_type="under")
    elif under_method == "tomek":
        return TomekLinksCUDA(n_jobs=n_jobs)
    else:
        raise ValueError(f"Unknown undersampling method: {under_method}")


# # Step 5.3: HPO (execution)

# In[41]:

# TODO IMPORTANT !!!
y_train_filtered = cudf.Series(y_train_filtered)

import optuna

# Define objective function for Optuna HPO
def objective(trial):
    try:

        # Apply Data Balancing

        assert isinstance(X_train_filtered, cudf.DataFrame)
        assert isinstance(y_train_filtered, cudf.Series)

        over_method = trial.suggest_categorical('over_method', over_method_choices)
        over_threshold = trial.suggest_categorical('over_threshold', over_threshold_choices)
        under_method = trial.suggest_categorical('under_method', under_method_choices)
        under_threshold = trial.suggest_categorical('under_threshold', under_threshold_choices)

        X_dtypes_before = X_train_filtered.dtypes.copy(deep=True).sort_index()
        y_dtype_before = y_train_filtered.dtype

        X_train_resampled, y_train_resampled = fit_resample(
            X_train_filtered, y_train_filtered, over_method, over_threshold, under_method, under_threshold
        )

        # TODO add optional sample weighting
        
        X_dtypes_after = X_train_resampled.dtypes.copy(deep=True).sort_index()
        y_dtype_after = y_train_resampled.dtype

        assert isinstance(X_train_resampled, cudf.DataFrame)
        assert isinstance(y_train_resampled, cudf.Series)
        assert X_dtypes_before.equals(X_dtypes_after), "X dtypes mismatch after resampling"
        assert y_dtype_before == y_dtype_after, "y dtype mismatch after resampling"
        assert X_train_resampled.shape[0] == y_train_resampled.shape[0]
        assert X_train_resampled.shape[1] == X_train_filtered.shape[1]
        assert X_train_resampled.isna().sum().sum() == 0
        assert y_train_resampled.isna().sum().sum() == 0
        assert set(y_train_resampled.to_pandas().unique()) == set(df_full['label'].to_pandas().unique())

        # Classifier hyperparameters (to be tuned)
        hpo_booster_params = {
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),                # Learning rate, controls step size at each iteration
            "max_depth": trial.suggest_int("max_depth", 3, 8),                     # Maximum depth of trees, prevents overfitting
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),      # Minimum sum of instance weight needed in a child node
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),               # Fraction of training data used per boosting round
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), # Fraction of features used per tree
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),         # L2 regularization term (Ridge regression)
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),           # L1 regularization term (Lasso regression)
            "gamma": trial.suggest_float("gamma", 0, 5.0)                          # Minimum loss reduction to make a split
        }
        print('here')

        # Perform cross-validation using train_xgb function with custom booster params
        model_hpo, train_time_hpo, latency_hpo, f1_weighted_hpo, model_size_hpo = \
            train_xgb(
                X_train_resampled, None, X_val_reduced,   # No validation set since CV == True
                y_train_resampled, None, y_val_sampled,   # No validation set since CV == True
                cv=True,                                  # Enable cross-validation
                custom_booster_params=hpo_booster_params, # Pass the tuned parameters
                verbose=0,                                # Suppress CV output during HPO
                persist=False                             # Do not persist model during HPO
            )

        return latency_hpo, f1_weighted_hpo

    except (ValueError, Exception) as e:
        # print(e)
        # if 'explicitly construct a GPU matrix' in str(e):
        #     pass
        raise optuna.TrialPruned(
            f'Over_Method: {trial.params["over_method"]} | Over_Threshold: {trial.params["over_threshold"]} | '
            f'Under_Method: {trial.params["under_method"]} | Under_Threshold: {trial.params["under_threshold"]} => {str(e)}'
        )


# In[ ]:


# Run Optuna optimization
start_time = time.time()
sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=random_state)
study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
study.optimize(objective, n_trials=hpo_n_trials, timeout=hpo_timeout)
hpo_time = time.time() - start_time
print(f"Execution Time (HPO): {hpo_time:.3f} seconds")


# In[ ]:


col_order = ['number', 'datetime_start', 'datetime_complete', 'duration', 'latency', 'f1_score',
             "over_method", "over_threshold", "under_method", "under_threshold",
             "eta", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "lambda", "alpha"]

df_hpo = study.trials_dataframe()

df_hpo_styled = df_hpo \
    .rename(columns={'values_0': 'latency', 'values_1': 'f1_score'}) \
    .rename(columns={c : c.replace('params_', '') for c in df_hpo.columns})[col_order].style \
    .background_gradient(cmap="RdYlGn_r", subset=["latency"]) \
    .background_gradient(cmap="RdYlGn", subset=["f1_score"]) \
    .format({"latency": "{:.2e}", "f1_score": "{:.6f}"}) \
    .set_table_styles([{"selector": "th, td", "props": "border: 1px solid black; text-align: center;"}])

df_hpo_styled.to_excel(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_trials.xlsx")
with open(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_trials.html", "w", encoding="utf-8") as f:
    f.write(df_hpo_styled.to_html())

df_hpo_styled


# In[ ]:

import plotly.graph_objects as go

fig = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0], target_name="Latency")
fig = go.Figure(fig)
fig.update_layout(width=None, height=None, autosize=True)
fig.write_html(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_hpo_optimization_history_1_latency.html", full_html=True)
# fig.show()


# In[ ]:


fig = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[1], target_name="Weighted F1 Score")
fig = go.Figure(fig)
fig.update_layout(width=None, height=None, autosize=True)
fig.write_html(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_hpo_optimization_history_2_score.html", full_html=True)
# fig.show()


# In[ ]:


def get_pareto_front(points):
    """Extracts the Pareto front from a set of (latency, F1-score) points."""
    pareto_front = []
    points = sorted(points, key=lambda x: x[0])  # Sort by latency (x-axis)

    max_f1 = -np.inf
    for x, y in points:
        if y > max_f1:  # A better (higher) F1-score found
            pareto_front.append((x, y))
            max_f1 = y

    return np.array(pareto_front)

# Generate Pareto front plot
fig = optuna.visualization.plot_pareto_front(study, target_names=["Latency", "Weighted F1 Score"])
fig = go.Figure(fig)

# Extract study results (Latency = values[0], F1 Score = values[1])
all_points = np.array([(t.values[0], t.values[1]) for t in study.trials if t.values])

# Get only the Pareto-optimal points
pareto_points = get_pareto_front(all_points)

# Add Pareto front as a semi-transparent line
fig.add_trace(
    go.Scatter(
        x=pareto_points[:, 0], 
        y=pareto_points[:, 1], 
        mode="lines",
        line=dict(color="red", width=3, dash="solid"),
        opacity=0.2,  # Set opacity (0 = fully transparent, 1 = fully opaque)
        name="Pareto Front"
    )
)

fig.update_layout(
    legend=dict(
        x=1.05,  # Horizontal position (0 = left, 1 = right)
        y=1.15,  # Vertical position (0 = bottom, 1 = top)
        bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
        bordercolor="black",  # Optional: Add a border
        borderwidth=1
    )
)

# Invert x-axis so lower latency is on the right
fig.update_layout(xaxis=dict(autorange="reversed"), width=None, height=None, autosize=True)

# Save and show plot
fig.write_html(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_hpo_optimization_history_3_balance.html", full_html=True)
# fig.show()


# In[ ]:


fig = optuna.visualization.plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="Latency")
fig = go.Figure(fig)
fig.update_layout(width=None, height=None, autosize=True)
fig.write_html(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_hpo_parallel_coordinate_1_latency.html", full_html=True)
# fig.show()


# In[ ]:


fig = optuna.visualization.plot_parallel_coordinate(study, target=lambda t: t.values[1], target_name="Weighted F1 Score")
fig = go.Figure(fig)
fig.update_layout(width=None, height=None, autosize=True)
fig.write_html(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_hpo_parallel_coordinate_2_score.html", full_html=True)
# fig.show()


# In[ ]:


if plot_param_importances:

    from plotly.subplots import make_subplots

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Latency", "Weighted F1 Score"])
    fig = go.Figure(fig)

    # Generate parameter importance plot for Latency
    fig_latency = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="Latency"
    )
    fig_latency.data[0].name = "Latency"  # Rename trace
    fig.add_trace(fig_latency.data[0], row=1, col=1)

    # Generate parameter importance plot for Weighted F1 Score
    fig_f1_score = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[1], target_name="Weighted F1 Score"
    )
    fig_f1_score.data[0].name = "Weighted F1 Score"  # Rename trace
    fig.add_trace(fig_f1_score.data[0], row=1, col=2)

    # Update layout
    fig.update_layout(title_text="Hyperparameter Importance for F1 Score & Latency", width=None, height=None, autosize=True)

    # Save and show plot
    fig.write_html(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_hpo_param_importances.html", full_html=True)
    # fig.show()


# # Step 5.4a: Updated Evaluation (with Sampling, Feature Selection, Row Filtering, and HPO) => Runner

# In[ ]:


import joblib
from copy import deepcopy

def reconstruct_and_evaluate_best_model(best_params, optimized_metric, persist: bool = True, filename: str = "deploy/multiclass/xgb_model.json"):

    # Copy to prevent modifying the original dictionary
    best_params_copy = deepcopy(best_params)

    # Redo resampling
    best_over_method = best_params_copy.pop('over_method')
    best_over_threshold = best_params_copy.pop('over_threshold')
    best_under_method = best_params_copy.pop('under_method')
    best_under_threshold = best_params_copy.pop('under_threshold')

    X_train_resampled, y_train_resampled = fit_resample(
        X_train_filtered, y_train_filtered, best_over_method, best_over_threshold, best_under_method, best_under_threshold
    )

    # TODO add optional sample weighting
    # TODO check the subsets !!!

    # Train final model using train_xgb function
    model_best, train_time_best, latency_best, f1_weighted_best, model_size_best = train_xgb(
        X_train_resampled, X_val_reduced, X_test_reduced,
        y_train_resampled, y_val_sampled, y_test_sampled,
        cv=False,  # No cross-validation, training final model
        custom_booster_params=best_params_copy,
        persist=persist,
        filename=filename
    )

    # Print results
    print(f"Training Time (best {optimized_metric}): {train_time_best:.3f} seconds")
    print(f"Latency (best {optimized_metric}): {latency_best:.2e} seconds")
    print(f"Weighted F1-Score (best {optimized_metric}): {f1_weighted_best:.6f}")
    print(f"Model Size (best {optimized_metric}): {model_size_best} MB")

    return X_train_resampled, model_size_best, train_time_best, latency_best, f1_weighted_best, best_params


# # Step 5.4b: Updated Evaluation (with Sampling, Feature Selection, Row Filtering, and HPO) => Lowest Latency

# In[ ]:


# Select best trial based on chosen strategy (here, minimizing Latency)
optimized_metric = 'Latency'
best_params = min(study.best_trials, key=lambda t: t.values[0]).params
print(f"Best Hyperparameters for {optimized_metric}:", best_params)

# Reconstruct and evaluate the best model
X_train_resampled_best_latency, model_size_best_latency, train_time_best_latency, latency_best_latency, f1_weighted_best_latency, best_params_best_latency = \
    reconstruct_and_evaluate_best_model(best_params, optimized_metric, True, "deploy/multiclass/xgb_best_latency.json")


# # Step 5.4c: Updated Evaluation (with Sampling, Feature Selection, Row Filtering, and HPO) => Highest Weighted F1 Score

# In[ ]:


# Select best trial based on chosen strategy (here, maximizing weighted F1 Score)
optimized_metric = 'Weighted F1 Score'
best_params = max(study.best_trials, key=lambda t: t.values[1]).params
print(f"Best Hyperparameters for {optimized_metric}:", best_params)

# Reconstruct and evaluate the best model
X_train_resampled_best_f1_score, model_size_best_f1_score, train_time_best_f1_score, latency_best_f1_score, f1_weighted_best_f1_score, best_params_best_f1_score = \
    reconstruct_and_evaluate_best_model(best_params, optimized_metric, True, "deploy/multiclass/xgb_best_f1_score.json")


# # Step 5.4d: Updated Evaluation (with Sampling, Feature Selection, Row Filtering, and HPO) => Best Balance

# In[ ]:


# Select best trial based on chosen strategy (here, maximizing weighted F1 Score)
optimized_metric = 'Balance'

# Extract Pareto-optimal trials
pareto_trials = study.best_trials  

# Normalize F1-score and Latency across Pareto-optimal trials
latency_values = np.array([t.values[0] for t in pareto_trials])  # Latency (lower is better)
f1_values = np.array([t.values[1] for t in pareto_trials])  # F1-score (higher is better)

# Min-max normalization for Latency (avoid division by zero, inverted since lower is better)
latency_min, latency_max = latency_values.min(), latency_values.max()
if latency_max - latency_min == 0:
    latency_norm = np.ones_like(latency_values)  # Assign 1 if all values are the same
else:
    latency_norm = (latency_max - latency_values) / (latency_max - latency_min)  # Inverted

# Min-max normalization for F1-score (avoid division by zero)
f1_min, f1_max = f1_values.min(), f1_values.max()
if f1_max - f1_min == 0:
    f1_norm = np.ones_like(f1_values)  # Assign 1 if all values are the same
else:
    f1_norm = (f1_values - f1_min) / (f1_max - f1_min)

# Compute a balanced trade-off score (equal weight for F1 and latency)
trade_off_scores = 0.5 * f1_norm + 0.5 * latency_norm

# Select the best trial based on the highest trade-off score
best_tradeoff_trial = pareto_trials[np.argmax(trade_off_scores)]

# Extract best hyperparameters
best_params = best_tradeoff_trial.params
print(f"Best Hyperparameters for {optimized_metric}:", best_params)

# Reconstruct and evaluate the best model
X_train_resampled_best_balance, model_size_best_balance, train_time_best_balance, latency_best_balance, f1_weighted_best_balance, best_params_best_balance = \
    reconstruct_and_evaluate_best_model(best_params, optimized_metric, True, "deploy/multiclass/xgb_best_balance.json")


# # Step 6: Summary

# In[ ]:


total_trials = len(study.trials)
completed_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
completion_ratio_str = f"{completed_trials}/{total_trials} ({hpo_n_trials})"


# In[ ]:


def round_floats_in_dict(d, decimals=3):
    return {
        key: (
            round_floats_in_dict(value, decimals) if isinstance(value, dict)  # Recursively process dicts
            else round(value, decimals) if isinstance(value, float)  # Round floats
            else value  # Keep everything else unchanged
        )
        for key, value in d.items()
    }


# In[ ]:


summary = {
    'X_train_shape': {
        'full': X_train_full.shape,
        'sampled': X_train_sampled.shape,
        'reduced': X_train_reduced.shape,
        'filtered': X_train_filtered.shape,
        'hpo_best_latency': X_train_resampled_best_latency.shape,
        'hpo_best_f1_score': X_train_resampled_best_f1_score.shape,
        'hpo_best_balance': X_train_resampled_best_balance.shape
    },
    'X_val_shape': {
        'full': X_val_full.shape,
        'sampled': X_val_sampled.shape,
        'reduced': X_val_reduced.shape,
        'filtered': X_val_reduced.shape,
        'hpo_best_latency': X_val_reduced.shape,
        'hpo_best_f1_score': X_val_reduced.shape,
        'hpo_best_balance': X_val_reduced.shape
    },
    'X_test_shape': {
        'full': X_test_full.shape,
        'sampled': X_test_sampled.shape,
        'reduced': X_test_reduced.shape,
        'filtered': X_test_reduced.shape,
        'hpo_best_latency': X_test_reduced.shape,
        'hpo_best_f1_score': X_test_reduced.shape,
        'hpo_best_balance': X_test_reduced.shape
    },
    'model_size_abs': {
        'full': round(model_size_full, 2),
        'sampled': round(model_size_sampled, 2),
        'reduced': round(model_size_reduced, 2),
        'filtered': round(model_size_filtered, 2),
        'hpo_best_latency': round(model_size_best_latency, 2),
        'hpo_best_f1_score': round(model_size_best_f1_score, 2),
        'hpo_best_balance': round(model_size_best_balance, 2)
    },
    'model_size_delta': {
        'full': f"{model_size_full - model_size_full:+.2f} ({(model_size_full - model_size_full) / model_size_full * 100:+.2f}%)",
        'sampled': f"{model_size_sampled - model_size_full:+.2f} ({(model_size_sampled - model_size_full) / model_size_full * 100:+.2f}%)",
        'reduced': f"{model_size_reduced - model_size_full:+.2f} ({(model_size_reduced - model_size_full) / model_size_full * 100:+.2f}%)",
        'filtered': f"{model_size_filtered - model_size_full:+.2f} ({(model_size_filtered - model_size_full) / model_size_full * 100:+.2f}%)",
        'hpo_best_latency': f"{model_size_best_latency - model_size_full:+.2f} ({(model_size_best_latency - model_size_full) / model_size_full * 100:+.2f}%)",
        'hpo_best_f1_score': f"{model_size_best_f1_score - model_size_full:+.2f} ({(model_size_best_f1_score - model_size_full) / model_size_full * 100:+.2f}%)",
        'hpo_best_balance': f"{model_size_best_balance - model_size_full:+.2f} ({(model_size_best_balance - model_size_full) / model_size_full * 100:+.2f}%)"
    },
    'training_time_abs': {
        'full': round(train_time_full, 3),
        'sampled': round(train_time_sampled, 3),
        'reduced': round(train_time_reduced, 3),
        'filtered': round(train_time_filtered, 3),
        'hpo_best_latency': round(train_time_best_latency, 3),
        'hpo_best_f1_score': round(train_time_best_f1_score, 3),
        'hpo_best_balance': round(train_time_best_balance, 3)
    },
    'training_time_delta': {
        'full': "--",
        'sampled': f"{train_time_sampled - train_time_full:+.6f} ({(train_time_sampled - train_time_full) / train_time_full * 100:+.2f}%)",
        'reduced': f"{train_time_reduced - train_time_full:+.6f} ({(train_time_reduced - train_time_full) / train_time_full * 100:+.2f}%)",
        'filtered': f"{train_time_filtered - train_time_full:+.6f} ({(train_time_filtered - train_time_full) / train_time_full * 100:+.2f}%)",
        'hpo_best_latency': f"{train_time_best_latency - train_time_full:+.6f} ({(train_time_best_latency - train_time_full) / train_time_full * 100:+.2f}%)",
        'hpo_best_f1_score': f"{train_time_best_f1_score - train_time_full:+.6f} ({(train_time_best_f1_score - train_time_full) / train_time_full * 100:+.2f}%)",
        'hpo_best_balance': f"{train_time_best_balance - train_time_full:+.6f} ({(train_time_best_balance - train_time_full) / train_time_full * 100:+.2f}%)"
    },
    'latency_abs': {
        'full': round(latency_full, 12),
        'sampled': round(latency_sampled, 12),
        'reduced': round(latency_reduced, 12),
        'filtered': round(latency_filtered, 12),
        'hpo_best_latency': round(latency_best_latency, 12),
        'hpo_best_f1_score': round(latency_best_f1_score, 12),
        'hpo_best_balance': round(latency_best_balance, 12)
    },
    'latency_delta': {
        'full': "--",
        'sampled': f"{latency_sampled - latency_full:+.2e} ({(latency_sampled - latency_full) / latency_full * 100:+.2f}%)",
        'reduced': f"{latency_reduced - latency_full:+.2e} ({(latency_reduced - latency_full) / latency_full * 100:+.2f}%)",
        'filtered': f"{latency_filtered - latency_full:+.2e} ({(latency_filtered - latency_full) / latency_full * 100:+.2f}%)",
        'hpo_best_latency': f"{latency_best_latency - latency_full:+.2e} ({(latency_best_latency - latency_full) / latency_full * 100:+.2f}%)",
        'hpo_best_f1_score': f"{latency_best_f1_score - latency_full:+.2e} ({(latency_best_f1_score - latency_full) / latency_full * 100:+.2f}%)",
        'hpo_best_balance': f"{latency_best_balance - latency_full:+.2e} ({(latency_best_balance - latency_full) / latency_full * 100:+.2f}%)"
    },
    'f1_score_abs': {
        'full': round(f1_weighted_full, 6),
        'sampled': round(f1_weighted_sampled, 6),
        'reduced': round(f1_weighted_reduced, 6),
        'filtered': round(f1_weighted_filtered, 6),
        'hpo_best_latency': round(f1_weighted_best_latency, 6),
        'hpo_best_f1_score': round(f1_weighted_best_f1_score, 6),
        'hpo_best_balance': round(f1_weighted_best_balance, 6)
    },
    'f1_score_delta': {
        'full': "--",
        'sampled': f"{f1_weighted_sampled - f1_weighted_full:+.6f} ({(f1_weighted_sampled - f1_weighted_full) * 100:+.2f}%)",
        'reduced': f"{f1_weighted_reduced - f1_weighted_full:+.6f} ({(f1_weighted_reduced - f1_weighted_full) * 100:+.2f}%)",
        'filtered': f"{f1_weighted_filtered - f1_weighted_full:+.6f} ({(f1_weighted_filtered - f1_weighted_full) * 100:+.2f}%)",
        'hpo_best_latency': f"{f1_weighted_best_latency - f1_weighted_full:+.6f} ({(f1_weighted_best_latency - f1_weighted_full) * 100:+.2f}%)",
        'hpo_best_f1_score': f"{f1_weighted_best_f1_score - f1_weighted_full:+.6f} ({(f1_weighted_best_f1_score - f1_weighted_full) * 100:+.2f}%)",
        'hpo_best_balance': f"{f1_weighted_best_balance - f1_weighted_full:+.6f} ({(f1_weighted_best_balance - f1_weighted_full) * 100:+.2f}%)"
    },
    'hpo_n_trials': {
        'full': "--",
        'sampled': "--",
        'reduced': "--",
        'filtered': "--",
        'hpo_best_latency': completion_ratio_str,
        'hpo_best_f1_score': completion_ratio_str,
        'hpo_best_balance': completion_ratio_str
    },
    'hpo_params': {
        'full': "--",
        'sampled': "--",
        'reduced': "--",
        'filtered': "--",
        'hpo_best_latency': round_floats_in_dict(best_params_best_latency),
        'hpo_best_f1_score': round_floats_in_dict(best_params_best_f1_score),
        'hpo_best_balance': round_floats_in_dict(best_params_best_balance)
    }
}

import pandas as pd

summary_df = pd.DataFrame(summary).style \
    .background_gradient(cmap="RdYlGn_r", subset=["model_size_abs"]) \
    .background_gradient(cmap="RdYlGn_r", subset=["training_time_abs"]) \
    .background_gradient(cmap="RdYlGn_r", subset=["latency_abs"]) \
    .background_gradient(cmap="RdYlGn", subset=["f1_score_abs"]) \
    .format({
        "model_size_abs": "{:.2f}",
        "training_time_abs": "{:.3f}",
        "f1_score_abs": "{:.6f}",
        "latency_abs": "{:.2e}",
    }) \
    .set_table_styles([
        {"selector": "th, td", "props": "border: 1px solid black; text-align: center;"}
    ])

summary_df.to_excel(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_summary_table.xlsx")
with open(f"output/nc={len(labels)}_sr={sampling_rate_sets:.2f}_summary_table.html", "w", encoding="utf-8") as f:
    f.write(summary_df.to_html())

summary_df


# In[ ]:


print(f"Global Time: {time.time() - global_start:.3f} seconds")

