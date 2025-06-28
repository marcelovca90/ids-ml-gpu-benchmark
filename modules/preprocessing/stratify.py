    def ensure_min_samples_per_class(df, stratify_col, min_samples_per_class):
        """Ensures that each class has at least `min_samples_per_class` samples using oversampling."""
        class_counts = df[stratify_col].value_counts()

        # Oversample minority classes
        oversampled = [
            df[df[stratify_col] == c].sample(n=min_samples_per_class, replace=True)
            for c in class_counts[class_counts < min_samples_per_class].index
        ]
        df_oversampled = pd.concat([df] + oversampled, ignore_index=True) if oversampled else df

        return df_oversampled.reset_index(drop=True)  # Fix index mismatch

    def stratified_sample_fraction(df, stratify_col, fraction=0.1):
        return df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=42)
        )