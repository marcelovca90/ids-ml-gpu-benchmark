from sklearn.model_selection import train_test_split as skl_train_test_split
import pandas as pd

def ensure_min_samples_per_class(df, stratify_col, min_samples_per_class, random_state):
    """Ensures that each class has at least `min_samples_per_class` samples using oversampling."""
    class_counts = df[stratify_col].value_counts()

    # Oversample minority classes
    oversampled = [
        df[df[stratify_col] == c].sample(n=min_samples_per_class, replace=True, random_state=random_state)
        for c in class_counts[class_counts < min_samples_per_class].index
    ]
    df_oversampled = pd.concat([df] + oversampled, ignore_index=True) if oversampled else df

    return df_oversampled.reset_index(drop=True)  # Fix index mismatch

# def restore_dtypes(df, dtypes):
#     for col, dtype in dtypes.to_dict().items():
#         df[col] = df[col].astype(dtype)

def assign_subsets(df, stratify_col, train_frac, val_frac, test_frac, random_state):
    """Splits data into mutually exclusive train/val/test subsets before applying stratified sampling."""
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1"

    # Assign subset labels
    df_train, df_temp = skl_train_test_split(df, test_size=(1 - train_frac), stratify=df[stratify_col], random_state=random_state)
    df_val, df_test = skl_train_test_split(df_temp, test_size=(test_frac / (val_frac + test_frac)), stratify=df_temp[stratify_col], random_state=random_state)

    df_train["subset"] = "train"
    df_val["subset"] = "val"
    df_test["subset"] = "test"

    return pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

# def sample_group(x):
#     """Helper function to stratify sample while ensuring class presence."""
#     n_samples = max(min_samples_per_class, int(len(x) * sampling_rate_sets))
#     return x.sample(n=n_samples, replace=len(x) < n_samples, random_state=random_state)

# def stratified_sample(df, stratify_col, sample_sets, sampling_rate_sets, min_samples_per_class, random_state):
#     """Applies stratified sampling while ensuring minimum samples per class, only for selected subsets."""

#     # Apply stratified sampling only for the requested subsets
#     df_sampled = df.groupby(["subset", stratify_col], group_keys=False).apply(
#         lambda x: sample_group(x) if x["subset"].iloc[0] in sample_sets else x
#     )

#     return df_sampled.reset_index(drop=True)

def train_val_test_split_fn(df_full, target_column, min_samples_per_class, random_state):
    # Ignore helper cols when capturing baseline dtypes
    df_dtypes_before = (
        df_full
        .drop(columns=["_ROW_ID"], errors="ignore")
        .dtypes.copy(deep=True)
        .sort_index()
    )

    # Ensure enough samples before splitting
    split_ok = False
    while not split_ok:
        df_full = ensure_min_samples_per_class(df_full, target_column, min_samples_per_class, random_state)
        df_full = df_full.reset_index(drop=True)  # Reset index before splitting
        df_full["_ROW_ID"] = df_full.index.astype("int64")
        df_dtypes_backup = df_full.dtypes.copy(deep=True)

        # Factorize for stratification
        category_mappings = {}

        for col, dtype in df_full.dtypes.to_dict().items():
            if dtype == 'category':
                codes, uniques = df_full[col].factorize()
                df_full[col] = codes.astype('int32')
                category_mappings[col] = uniques.tolist()

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

    # Ignore helper cols when comparing dtypes
    df_dtypes_after = (
        df_full
        .drop(columns=["subset", "_ROW_ID"], errors="ignore")
        .dtypes.copy(deep=True)
        .sort_index()
    )

    assert df_dtypes_before.equals(df_dtypes_after), "Dtype mismatch after assigning subsets"

    df_train = df_full[df_full["subset"] == "train"].drop(columns=["subset"])
    df_val = df_full[df_full["subset"] == "val"].drop(columns=["subset"])
    df_test = df_full[df_full["subset"] == "test"].drop(columns=["subset"])

    return df_train, df_val, df_test

def sample_train_subset(
    df_train_full: pd.DataFrame,
    frac: float = 1.0,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """
    Downsample the training DataFrame using fraction `frac`.
    If frac == 1.0, returns the full DataFrame unchanged.
    """

    if frac is None:
        frac = 1.0

    # No sampling needed
    if frac >= 1.0:
        return df_train_full.copy()

    if frac <= 0:
        raise ValueError("frac must be > 0")

    # Perform fractional sampling
    df_sampled = df_train_full.sample(frac=frac, random_state=random_state)

    return df_sampled.reset_index(drop=True)
