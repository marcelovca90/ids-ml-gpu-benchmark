import json
import numbers
import os
from collections import OrderedDict
from operator import getitem

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from featurewiz import FeatureWiz
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold, TomekLinks
from pytictoc import TicToc
from scipy import stats
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import ExtraTreeClassifier

from modules.logging.logger import log_print

t = TicToc()
JOBS = 4
SEED = 10


def _baseline_evaluation(df, label_column='label'):
    log_print(f"Performing baseline evaluation...")
    X_train, X_test, y_train, y_test = _train_test_split(df, label_column)
    cls = ExtraTreeClassifier(random_state=SEED)
    cls.fit(X_train, y_train)
    log_print(
        f"Shape: {df.shape} => Training Score: {cls.score(X_train, y_train)}")
    log_print(
        f"Shape: {df.shape} => Test Score    : {cls.score(X_test, y_test)}")


def _determine_best_chunksize(csv_file, sep, comment, target_memory_usage=0.8):
    # Get total number of rows in the CSV file
    total_rows = sum(1 for line in open(csv_file))
    # Calculate approximate memory usage for a single chunk with one row
    sample_chunk = pd.read_csv(csv_file, sep=sep, comment=comment, nrows=1)
    memory_per_row = sample_chunk.memory_usage(deep=True).sum() / 1024**2
    # Get available system memory
    available_memory = psutil.virtual_memory().available
    available_memory = available_memory / (1024**2)
    # Calculate the chunksize based on available memory and target memory usage
    chunksize = int((available_memory * target_memory_usage) / memory_per_row)
    return min(chunksize, total_rows)


def _convert_to_int(x):
    try:
        if '.' in x:
            return int(float(x))
        elif '0x' in x:
            return int(x, 16)
        else:
            return int(x)
    except:
        return 0


def _downcast_dtypes(df: pd.DataFrame, inplace=False):
    log_print('Downcasting numeric types to save memory...')
    _df = df if inplace else df.copy()
    for col in _df.columns:
        # integers
        if issubclass(_df[col].dtypes.type, numbers.Integral):
            # unsigned integers
            if _df[col].min() >= 0:
                _df[col] = pd.to_numeric(_df[col], downcast='unsigned')
            # signed integers
            else:
                _df[col] = pd.to_numeric(_df[col], downcast='integer')
        # other real numbers
        elif issubclass(_df[col].dtypes.type, numbers.Real):
            _df[col] = pd.to_numeric(_df[col], downcast='float')
    if not inplace:
        return _df


def _drop_duplicates(df, inplace=False, create_count_column=False):
    _df = df if inplace else df.copy()
    if create_count_column:
        current_cols = [x for x in _df.columns]
        log_print(
            f'Duplicated rows before removal: {_df.duplicated().sum()}')
        _df['count'] = 1
        _df.groupby(current_cols)['count'].count(
        ).reset_index().drop_duplicates()
        log_print(
            f'Duplicated after before removal: {_df.duplicated().sum()}')
    else:
        log_print(f'Dropping {_df.duplicated().sum()} duplicated rows...')
        _df.drop_duplicates()
    if not inplace:
        return _df


def _drop_less_relevant_columns(df, label_column, threshold=0.1, inplace=False):
    _df = df if inplace else df.copy()
    X, y = _df.drop(columns=[label_column]), _df[label_column]
    constant_filter = VarianceThreshold(threshold=threshold).fit(X, y)
    support_columns = X.columns[constant_filter.get_support()]
    non_constant_columns = [
        col for col in _df.columns if col not in support_columns]
    log_print(f'Non (quasi-)constant columns: {non_constant_columns}')
    constant_columns = [col for col in _df.columns if col in support_columns]
    log_print(f'(Quasi-)constant columns: {constant_columns}')
    _df.drop(columns=constant_columns)
    if not inplace:
        return _df


def _filter_by_frequency(df, column, min_rel_freq_pct=0.01, inplace=False):
    _df = df if inplace else df.copy()
    log_print(f"Value counts before filtering by frequency:")
    _pretty_print_value_counts(_df, column)
    vcd = _df[column].value_counts(normalize=True).to_dict()
    relevant_labels = [key for key,
                       value in vcd.items() if value > min_rel_freq_pct/100.0]
    log_print(
        f'Dropping rows with relative frequency inferior to {min_rel_freq_pct:.3f}% ...')
    filtered_labels = _df[column].value_counts().index.drop(relevant_labels)
    for label in filtered_labels:
        _df = _df.drop(_df[_df.label == label].index)
    log_print(f"Value counts after filtering by frequency:")
    _pretty_print_value_counts(_df, column)
    if not inplace:
        return _df


def _filter_by_quantile(df, column, percentage=0.05):
    value_counts = df[column].value_counts()
    log_print(f"Value counts before filtering by quantile:")
    _pretty_print_value_counts(df, column)
    threshold = value_counts.quantile(percentage)
    log_print(
        f"Dropping '{column}' rows with less than {threshold:.2f} occurrences...")
    df = df[df[column].isin(value_counts.index[value_counts.ge(threshold)])]
    log_print(f"Value counts before filtering by quantile:")
    _pretty_print_value_counts(df, column)
    log_print(
        f'Filtering by quantile performed succesfully; new DF shape: {df.shape}.')
    return df


def _label_encode(df, column):
    encoder = LabelEncoder().fit(df[column])
    log_print(
        f"Label encoder found the following classes for '{column}': " +
        str(encoder.classes_))
    df[column] = encoder.transform(df[column])
    mappings = {}
    for _class in encoder.classes_:
        mappings.update({str(_class): str(encoder.transform([_class])[0])})
    return df.astype({column: np.uint8}), mappings


def _one_hot_encode(df, column):
    df = pd.get_dummies(df, columns=[column])
    log_print(
        f'Column \'{column}\' successfully one-hot-encoded; ' +
        f'new DF shape: {df.shape}.')
    return df


def _min_max_scale(df, column, inplace=False):
    _df = df if inplace else df.copy()
    log_print(
        f'Performing MinMax scaling on \'{column}\'; original range: [{_df[column].min()} - {_df[column].max()}]...')
    _df[[column]] = MinMaxScaler().fit_transform(_df[[column]])
    log_print(
        f'Column \'{column}\' successfully MinMax scaled; new range: [{_df[column].min()} - {_df[column].max()}].')
    return _df


def _persist_mappings(mappings, base_folder, filename='mappings.json'):
    log_print(f'Persisting mappings to \'{filename}\'...')
    full_filename = os.path.join(base_folder, filename)
    with open(full_filename, 'w') as fp:
        json.dump(mappings, fp)
    log_print(f'Mappings persisted to \'{filename}\'.')


def _persist_dataset(df, base_folder, file_name, formats=['csv']):
    log_print(f'Persisting dataset to \'{os.path.dirname(base_folder)}\'...')
    if 'csv' in formats:
        full_filename = os.path.join(base_folder, f'{file_name}.csv')
        df.round(3).to_csv(full_filename,
                           float_format='%g', header=None, index=None)
        log_print(f'CSV persisted to \'{file_name}.csv\'.')
    if 'ftr' in formats:
        full_filename = os.path.join(base_folder, f'{file_name}.ftr')
        df.round(3).reset_index(drop=True).to_feather(full_filename)
        log_print(f'FTR persisted to \'{file_name}.ftr\'.')
    if 'hdf' in formats:
        full_filename = os.path.join(base_folder, f'{file_name}.h5')
        df.round(3).to_hdf(full_filename, file_name)
        log_print(f'HDF persisted to \'{file_name}.h5\'.')
    if 'parquet' in formats:
        full_filename = os.path.join(base_folder, f'{file_name}.parquet')
        df.round(3).to_parquet(full_filename)
        log_print(f'Parquet persisted to \'{file_name}.parquet\'.')


def _persist_subsets(X_train, X_test, y_train, y_test, base_folder, formats=['csv']):
    log_print(f'Persisting subsets to \'{os.path.dirname(base_folder)}\'...')
    if 'csv' in formats:
        pd.DataFrame(X_train, copy=False).round(3).to_csv(os.path.join(
            base_folder, 'X_train.csv'), float_format='%g', header=None, index=None)
        pd.DataFrame(X_test, copy=False).round(3).to_csv(os.path.join(
            base_folder, 'X_test.csv'), float_format='%g', header=None, index=None)
        pd.DataFrame(y_train, copy=False).round(3).to_csv(os.path.join(
            base_folder, 'y_train.csv'), float_format='%g', header=None, index=None)
        pd.DataFrame(y_test, copy=False).round(3).to_csv(os.path.join(
            base_folder, 'y_test.csv'), float_format='%g', header=None, index=None)
        log_print(
            f'CSV subsets persisted to \'{os.path.dirname(base_folder)}\'.')
    if 'ftr' in formats:
        pd.DataFrame(X_train, copy=False).round(3).reset_index(
            drop=True).to_feather(os.path.join(base_folder, 'X_train.ftr'))
        pd.DataFrame(X_test, copy=False).round(3).reset_index(
            drop=True).to_feather(os.path.join(base_folder, 'X_test.ftr'))
        pd.DataFrame(y_train, copy=False).round(3).reset_index(
            drop=True).to_feather(os.path.join(base_folder, 'y_train.ftr'))
        pd.DataFrame(y_test, copy=False).round(3).reset_index(
            drop=True).to_feather(os.path.join(base_folder, 'y_test.ftr'))
        log_print(
            f'FTR subsets persisted to \'{os.path.dirname(base_folder)}\'.')
    if 'hdf' in formats:
        pd.DataFrame(X_train, copy=False).round(3).to_hdf(
            os.path.join(base_folder, 'X_train.h5'), 'X_train')
        pd.DataFrame(X_test, copy=False).round(3).to_hdf(
            os.path.join(base_folder, 'X_test.h5'), 'X_test',)
        pd.DataFrame(y_train, copy=False).round(3).to_hdf(
            os.path.join(base_folder, 'y_train.h5'), 'y_train')
        pd.DataFrame(y_test, copy=False).round(3).to_hdf(
            os.path.join(base_folder, 'y_test.h5'), 'y_test')
        log_print(
            f'HDF subsets persisted to \'{os.path.dirname(base_folder)}\'.')


def _plot_feature_importances(df, label_column, base_folder,):
    log_print(f"Calculating and plotting feature importances...")
    # get importances from a tree-based classifier
    X, y = df.drop(columns=[label_column]), df[label_column]
    cls = ExtraTreeClassifier(random_state=SEED)
    cls.fit(X, y)
    importances = cls.feature_importances_
    indices = np.argsort(importances)
    features = X.columns
    # plot importances for each feature
    base_filename = f'feature_importances_{len(features)}.png'
    full_filename = os.path.join(base_folder, base_filename)
    plt.clf()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(full_filename)
    log_print(f'Feature importance plots persisted to \'{base_filename}\'.')


def _pretty_print_value_counts(df, column, lpad=64, rpad=12):
    vc_norm_f = df[column].value_counts(normalize=False).to_dict()
    vc_norm_t = df[column].value_counts(normalize=True).to_dict()
    ans = {}
    for label in df[column].unique():
        ans[label] = {'abs': vc_norm_f[label], 'rel': vc_norm_t[label]}
    ans = OrderedDict(
        sorted(ans.items(), key=lambda x: getitem(x[1], 'abs'), reverse=True))
    output = []
    output.append(
        f"+-{'-----'.ljust(lpad,'-')}-+-{'-----'.rjust(rpad,'-')}-+-{'---------'.rjust(rpad,'-')}-+")
    output.append(
        f"| {column.center(lpad)} | {'Count'.center(rpad)} | {'Count (%)'.center(rpad)} |")
    output.append(
        f"+-{'-----'.ljust(lpad,'-')}-+-{'-----'.rjust(rpad,'-')}-+-{'---------'.rjust(rpad,'-')}-+")
    for key, value in ans.items():
        col_1 = f"{str(key).ljust(lpad)}"
        col_2 = f"{value['abs']}".rjust(rpad)
        col_3 = f"{(100.0 * value['rel']):.06f}".rjust(rpad)
        output.append(f'| {col_1} | {col_2} | {col_3} |')
    output.append(
        f"+-{'-----'.ljust(lpad,'-')}-+-{'-----'.rjust(rpad,'-')}-+-{'---------'.rjust(rpad,'-')}-+")
    for line in output:
        log_print(line)
    return output


def _replace_values(df, column, old_value, new_value):
    df.loc[(df[column] == old_value), column] = new_value


def _select_relevant_features(df, label_column, mode, n_folds=5, skip_sulov=False, variance_threshold=0.95):
    if mode == 'featurewiz':
        log_print(f'Performing feature selection with FeatureWiz...')
        X, y = df.drop(columns=[label_column]), df[label_column]
        wiz = FeatureWiz(corr_limit=0.90, skip_sulov=skip_sulov, verbose=0)
        wiz.fit(X, y)
        relevant_columns = [col for col in X.columns if col in wiz.features]
        log_print(f'Features that will be kept: {relevant_columns}')
        irrelevant_cols = [col for col in X.columns if col not in wiz.features]
        log_print(f'Features that will be dropped: {irrelevant_cols}')
        return df.drop(columns=irrelevant_cols)
    elif mode == 'ipca':
        log_print(
            f'Performing feature selection with Incremental PCA (variance_threshold={variance_threshold:y.2f})...')
        X, y = df.drop(columns=[label_column]), df[label_column]
        X_columns, X_scaled = X.columns, StandardScaler(
            copy=False).fit_transform(X, y)
        current_n_components = int(len(X_columns)/2)
        current_variance_ratio = 0.0
        while current_variance_ratio < variance_threshold:
            t.tic()
            current_n_components += 1
            ipca = IncrementalPCA(
                n_components=current_n_components, batch_size=50*len(X_columns)).fit(X_scaled)
            current_variance_ratio = np.max(
                np.cumsum(ipca.explained_variance_ratio_))
            log_print(
                f'n_components={current_n_components} -> variance_ratio={current_variance_ratio:.6f} (iteration took {t.tocvalue(True):.2f}s)')
        return pd.DataFrame(data=ipca.transform(X_scaled), columns=ipca.get_feature_names_out(), index=df.index, copy=False).assign(label=y)
    elif mode == 'pca':
        log_print(
            f'Performing feature selection with PCA (variance_threshold={variance_threshold})...')
        X, y = df.drop(columns=[label_column]), df[label_column]
        X_columns, X_scaled = X.columns, StandardScaler(
            copy=False).fit_transform(X, y)
        pca = PCA(n_components=variance_threshold,
                  svd_solver='full').fit(X_scaled)
        return pd.DataFrame(data=pca.transform(X_scaled), columns=pca.get_feature_names_out(), index=df.index, copy=False).assign(label=y)
    elif mode == 'rfecv':
        log_print(
            f'Performing {n_folds}-fold recursive feature elimination...')
        # X, y = df.drop(columns=[label_column]).select_dtypes(include='number'), df[label_column]
        X, y = df.drop(columns=[label_column]), df[label_column]
        rfecv = RFECV(estimator=Ridge(random_state=SEED),
                      cv=n_folds, n_jobs=JOBS, verbose=1)
        rfecv.fit(X, y)
        feature_mask = X.columns[rfecv.get_support()]
        relevant_columns = [
            col for col in X.columns if col not in feature_mask]
        log_print(f'Features that will be kept: {relevant_columns}')
        irrelevant_cols = [col for col in X.columns if col in feature_mask]
        log_print(f'Features that will be dropped: {irrelevant_cols}')
        return df.drop(columns=irrelevant_cols)


def _sort_columns(df, rightmost_columns):
    final_cols = [x for x in df.columns.values if x not in rightmost_columns]
    final_cols.extend(rightmost_columns)
    log_print(f'Columns sorted according to {final_cols}.')
    return df.reindex(columns=final_cols)


def _resample_by_instance_hardness(X, y):
    log_print(
        f"X and y shapes before resampling by Instance Hardness: {X.shape}; {y.shape}")
    X, y = InstanceHardnessThreshold(estimator=ExtraTreeClassifier(
        random_state=SEED), n_jobs=-1, random_state=SEED).fit_resample(X, y)
    log_print(
        f"X and y shapes after resampling by Instance Hardness: {X.shape}; {y.shape}")
    return X, y


def _resample_by_tomek_links(X, y):
    log_print(
        f"X and y shapes before resampling by Tomek's Links: {X.shape}; {y.shape}")
    X, y = TomekLinks(sampling_strategy='auto', n_jobs=-1).fit_resample(X, y)
    log_print(
        f"X and y shapes after resampling by Tomek's Links: {X.shape}; {y.shape}")
    return X, y


def _resample_by_smote_tomek(X, y):
    log_print(
        f"X and y shapes before resampling by SMOTE and Tomek's Links: {X.shape}; {y.shape}")
    smote = SMOTE(sampling_strategy="auto", n_jobs=-1, random_state=42)
    tomek = TomekLinks(sampling_strategy="auto", n_jobs=-1)
    X, y = SMOTETomek(sampling_strategy="auto", smote=smote,
                      tomek=tomek, n_jobs=-1, random_state=42).fit_resample(X, y)
    log_print(
        f"X and y shapes after resampling by SMOTE and Tomek's Links: {X.shape}; {y.shape}")
    return X, y


def _train_test_split(df, label_column, test_size=0.2, random_state=SEED, resampling_strategies=[]):
    log_print(
        'Splitting dataset into training (X, y) and test (X, y) subsets...')
    X, y = df.drop(columns=[label_column]), df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    if 'hardness' in resampling_strategies:
        X_train, y_train = _resample_by_instance_hardness(X_train, y_train)
    if 'tomek' in resampling_strategies:
        X_train, y_train = _resample_by_tomek_links(X_train, y_train)
    if 'smote_tomek' in resampling_strategies:
        X_train, y_train = _resample_by_smote_tomek(X_train, y_train)
    log_print(
        f'X_train shape: {X_train.shape}; y_train shape: {y_train.shape}; y_train unique values: {set(y_train)}')
    log_print(
        f'X_test shape: {X_test.shape}; y_test shape: {y_test.shape}; y_test unique values: {set(y_test)}')
    return X_train, X_test, y_train, y_test

# Define a function to determine the threshold based on the z-score


def _determine_threshold(data):
    # Calculate the unique value counts for the column
    unique_value_counts = data.nunique()

    # Calculate the z-score for the number of unique values
    z_score = stats.zscore([unique_value_counts])

    # Determine the threshold as the absolute z-score
    threshold = abs(z_score[0])

    return threshold

# Define a function to automatically determine the QuantileTransformer distribution for each column


def _determine_quantile_distribution(data):
    # Perform the Shapiro-Wilk normality test
    _, p_value = stats.shapiro(data)

    # Set the distribution based on the p-value of the normality test
    # You can adjust the significance level (e.g., 0.05) as needed
    if p_value > 0.05:
        return 'normal'
    else:
        return 'uniform'


def _elbow_method(data):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Initialize lists to store inertia (within-cluster sum of squares) and silhouette scores
    inertia_values = []
    silhouette_scores = []

    # Define the range of possible cluster numbers
    k_range = range(2, 11)

    # Loop through different cluster numbers
    for k in k_range:
        # Create a KMeans clustering model
        log_print(f'k={k}')
        kmeans = KMeans(n_clusters=k, random_state=0)

        # Fit the model to the data
        kmeans.fit(data.values.reshape(-1, 1))

        # Append the inertia and silhouette score to the lists
        inertia_values.append(kmeans.inertia_)

        if k > 1:  # Silhouette score requires at least 2 clusters
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(
                data.values.reshape(-1, 1), labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(None)

    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()
