from sklearn.model_selection import train_test_split
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
    df_train, df_temp = train_test_split(
        df, test_size=(1 - train_frac), stratify=df[stratify_col], random_state=random_state
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=(test_frac / (val_frac + test_frac)), stratify=df_temp[stratify_col], random_state=random_state
    )

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
    # ---------------------------------------------------------
    # 1. Setup & ID Generation
    # ---------------------------------------------------------
    df_full = df_full.reset_index(drop=True)
    if "_ROW_ID" not in df_full.columns:
        df_full["_ROW_ID"] = df_full.index.astype("int64")

    # Capture baseline dtypes
    df_dtypes_before = (
        df_full
        .drop(columns=["_ROW_ID"], errors="ignore")
        .dtypes.copy(deep=True)
        .sort_index()
    )

    # ---------------------------------------------------------
    # 2. Dynamic Stratified Split
    # ---------------------------------------------------------
    # Start with 3 because you cannot stratify 3-split a single sample.
    rare_threshold = 3
    split_successful = False
    
    df_train, df_val, df_test = None, None, None

    while not split_successful:
        # A. Filter rare classes based on current threshold
        class_counts = df_full[target_column].value_counts()
        rare_classes = class_counts[class_counts < rare_threshold].index
        
        df_rare = df_full[df_full[target_column].isin(rare_classes)]
        df_common = df_full[~df_full[target_column].isin(rare_classes)]

        # If df_common is empty, we can't split anything.
        if df_common.empty:
             print("Warning: All classes are considered rare. Moving everything to train.")
             df_train = df_full.copy()
             df_val = df_full.iloc[:0].copy()
             df_test = df_full.iloc[:0].copy()
             split_successful = True
             break

        try:
            # B. Try Split 1: Train (60%) vs Temp (40%)
            train_common, temp_common = train_test_split(
                df_common, 
                test_size=0.4, 
                stratify=df_common[target_column], 
                random_state=random_state
            )
            
            # C. Try Split 2: Val (20%) vs Test (20%)
            val_common, test_common = train_test_split(
                temp_common, 
                test_size=0.5, 
                stratify=temp_common[target_column], 
                random_state=random_state
            )
            
            # If we reach here, the split worked!
            split_successful = True
            
            if not df_rare.empty:
                print(f"Stratification successful with rare_threshold={rare_threshold}. "
                      f"Moved {len(df_rare)} rows (classes < {rare_threshold} samples) to Train.")

            # Reassemble
            df_train = pd.concat([train_common, df_rare]).reset_index(drop=True)
            df_val   = val_common.reset_index(drop=True)
            df_test  = test_common.reset_index(drop=True)

        except ValueError as e:
            # Sklearn error: "The least populated class... has only X members"
            # We increment the threshold so that specific class gets moved to 'df_rare' next time.
            rare_threshold += 1
            # Safety break to prevent infinite loops (though unlikely)
            if rare_threshold > 50:
                raise RuntimeError("Could not split dataset even with rare_threshold=50. Dataset might be too small or imbalanced.") from e

    # ---------------------------------------------------------
    # 3. Oversample ONLY Training Data
    # ---------------------------------------------------------
    print(f"Oversampling Training set (target min_samples={min_samples_per_class})...")
    df_train = ensure_min_samples_per_class(
        df_train, 
        target_column, 
        min_samples_per_class, 
        random_state
    )
    
    # ---------------------------------------------------------
    # 4. Dtype Restoration, Cleanup & Shuffling
    # ---------------------------------------------------------
    def restore_types(df, ref_dtypes):
        for col, dtype in ref_dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    df_train = restore_types(df_train, df_dtypes_before)
    df_val   = restore_types(df_val, df_dtypes_before)
    df_test  = restore_types(df_test, df_dtypes_before)

    # Mix the rare classes and oversampled duplicates into the general population
    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_val   = df_val.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_test  = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True)

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
