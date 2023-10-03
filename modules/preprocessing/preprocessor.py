

import json
import os
from abc import ABC, abstractmethod

import featuretools as ft
import pandas as pd
from category_encoders import OneHotEncoder
from fastai.tabular.all import df_shrink
from featurewiz import FeatureWiz
from pandas_dq import dq_report
from scipy import stats
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                   QuantileTransformer)
from typing_extensions import Self

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.stats import (log_data_types, log_memory_usage,
                                         log_value_counts)


class BasePreprocessingPipeline(ABC):

    def __init__(self) -> None:
        self.data: pd.DataFrame = None
        self.folder: str = None
        self.name: str = None
        self.target: str = None
        self.metadata: dict = dict()
        self.seed = 42
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None

    @function_call_logger
    def preload(self) -> None:
        base_path = os.path.join(os.getcwd(), self.folder, 'generated')
        log_print(f'Loading cached files from \'{base_path}\'...')
        self.data = pd.read_parquet(
            os.path.join(base_path, self.name + '.parquet'))
        with open(os.path.join(base_path, 'metadata.json')) as json_file:
            self.metadata = json.load(json_file)
        self.X_train = pd.read_parquet(
            os.path.join(base_path, 'X_train.parquet'))
        self.y_train = pd.read_parquet(
            os.path.join(base_path, 'y_train.parquet'))
        self.X_test = pd.read_parquet(
            os.path.join(base_path, 'X_test.parquet'))
        self.y_test = pd.read_parquet(
            os.path.join(base_path, 'y_test.parquet'))

    @abstractmethod
    def prepare(self) -> None:
        pass

    def analyze(self) -> None:
        dqr = dq_report(self.data, target=self.target,
                        html=True, csv_engine="pandas", verbose=1)
        log_print(dqr)

    @abstractmethod
    def load(self) -> None:
        pass

    def remove_na_duplicates(self) -> None:
        # Display the number of duplicates and NAs before cleaning
        num_duplicates_before = self.data.duplicated().sum()
        num_nas_before = self.data.isna().sum().sum()
        log_print(f"Duplicates before cleaning: {num_duplicates_before}")
        log_print(f"NAs before cleaning: {num_nas_before}")
        # Remove duplicates
        self.data.drop_duplicates(inplace=True)
        # Remove NAs
        self.data.dropna(inplace=True)
        # Display the number of duplicates and NAs after cleaning
        num_duplicates_after = self.data.duplicated().sum()
        num_nas_after = self.data.isna().sum().sum()
        log_print(f"Duplicates after cleaning: {num_duplicates_after}")
        log_print(f"NAs after cleaning: {num_nas_after}")

    @abstractmethod
    def sanitize(self) -> None:
        pass

    def set_dtypes(self) -> None:
        log_print('Data types before inference:')
        log_data_types(self.data)
        es = ft.EntitySet(id="es")
        es = es.add_dataframe(dataframe_name=self.name,
                              dataframe=self.data.drop(columns=[self.target]),
                              make_index=True,
                              index='idx')
        feature_defs = ft.dfs(entityset=es, features_only=True,
                              target_dataframe_name=self.name, verbose=True)
        self.numeric_cols = [
            x.column_name for x in feature_defs if x.column_schema.is_numeric]
        self.categorical_cols = [
            x.column_name for x in feature_defs if x.column_schema.is_categorical]
        self.ordinal_cols = [
            x.column_name for x in feature_defs if x.column_schema.is_ordinal]
        feature_types = {
            x.column_name: x.column_schema.logical_type.primary_dtype for x in feature_defs}
        self.data = self.data.astype(feature_types)
        log_print('Data types after inference:')
        log_data_types(self.data)

    @function_call_logger
    def filter(self) -> None:
        log_print(f"Value counts before filtering by frequency:")
        log_value_counts(self.data, self.target)
        vcd = self.data[self.target].value_counts(normalize=True).to_dict()
        kept_labels = [key for key, val in vcd.items() if val > 1.0/100.0]
        log_print(f'Dropping labels with frequency inferior to 0.01% ...')
        filtered_labels = self.data[self.target].value_counts(
        ).index.drop(kept_labels)
        for label in filtered_labels:
            self.data = self.data.drop(
                self.data[self.data[self.target] == label].index)
        self.data.reset_index(drop=True, inplace=True)
        log_print(f"Value counts after filtering by frequency:")
        log_value_counts(self.data, self.target)

    @function_call_logger
    def encode(self) -> None:

        for col in self.numeric_cols + self.categorical_cols:
            num_unique = self.data[col].nunique()
            # OneHotEncode categorical columns
            if num_unique <= 10:
                encoder = OneHotEncoder(cols=[col], use_cat_names=True)
                self.data = encoder.fit_transform(self.data)
                log_print(
                    f'{col}({num_unique}) => OneHot()')
            # QuantileTransform numeric columns
            else:
                # Perform the Shapiro-Wilk normality test
                _, p_value = stats.shapiro(self.data[col])
                # Set the distribution based on the p-value of the normality test
                output_distribution = 'normal' if p_value > 0.05 else 'normal'
                transformer = QuantileTransformer(
                    output_distribution=output_distribution)
                self.data[col] = transformer.fit_transform(
                    self.data[col].values.reshape(-1, 1))
                log_print(
                    f'{col}({num_unique}) => Quantile({output_distribution})')

        # OrdinalEncode ordinal columns
        for col in self.ordinal_cols:
            encoder = OrdinalEncoder()
            self.data[col] = encoder.fit_transform(self.data[col])
            log_print(f'{col} => Ordinal ({set(self.data[col].unique())})')

        # TargetEncode target column
        encoder = LabelEncoder()
        self.data[self.target] = encoder.fit_transform(self.data[self.target])
        log_print(
            f'{self.target} => Label ({set(self.data[self.target].unique())})')

    @function_call_logger
    def reset_index(self) -> None:
        self.data.reset_index(drop=True, inplace=True)

    @function_call_logger
    def sort_columns(self) -> None:
        cols = [x for x in self.data.columns.values if x not in [self.target]]
        cols.extend([self.target])
        log_print(f'Columns sorted according to {cols}.')
        self.data = self.data.reindex(columns=cols)

    @function_call_logger
    def shrink_dtypes(self) -> None:
        log_print(f"Data types and memory usage before shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)
        self.data = df_shrink(self.data, obj2cat=True, int2uint=False)
        log_print(f"Data types and memory usage after shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def select_features(self) -> None:
        log_print(f'Performing feature selection with FeatureWiz...')
        X, y = self.data.drop(columns=[self.target]), self.data[self.target]
        wiz = FeatureWiz(corr_limit=0.90, skip_sulov=False, verbose=0)
        wiz.fit(X, y)
        relevant_columns = [col for col in X.columns if col in wiz.features]
        log_print(f'Features that will be kept: {str(relevant_columns)}')
        irrelevant_cols = [col for col in X.columns if col not in wiz.features]
        log_print(f'Features that will be dropped: {str(irrelevant_cols)}')
        return self.data.drop(columns=irrelevant_cols)

    @function_call_logger
    def train_test_split(self) -> None:
        X, y = self.data.drop(columns=[self.target]), self.data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=self.seed)

    @function_call_logger
    def update_metadata(self) -> None:
        self.metadata['columns'] = self.data.columns.to_list()
        self.metadata['description'] = self.data.describe().to_dict()
        self.metadata['dtypes'] = self.data.dtypes.to_dict()
        self.metadata['shape'] = self.data.shape
        self.metadata['memory_usage'] = self.data.memory_usage(deep=True).sum()
        self.metadata['value_counts'] = \
            self.data[self.target].value_counts().to_dict()
        self.metadata['X_train_shape'] = self.X_train.shape
        self.metadata['y_train_shape'] = self.y_train.shape
        self.metadata['X_test_shape'] = self.X_test.shape
        self.metadata['y_test_shape'] = self.y_test.shape

    @function_call_logger
    def save(self) -> None:
        # Dataset
        dataset_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', self.name + '.parquet')
        log_print(f'Persisting dataset to \'{dataset_filename}\'...')
        self.data.to_parquet(path=dataset_filename, index=False)
        # Train data
        X_train_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'X_train.parquet')
        log_print(f'Persisting X_train to \'{X_train_filename}\'...')
        self.X_train.to_parquet(path=X_train_filename, index=False)
        y_train_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'y_train.parquet')
        log_print(f'Persisting y_train to \'{y_train_filename}\'...')
        self.y_train.to_frame().to_parquet(path=y_train_filename, index=False)
        # Test data
        X_test_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'X_test.parquet')
        log_print(f'Persisting X_test to \'{X_test_filename}\'...')
        self.X_test.to_parquet(path=X_test_filename, index=False)
        y_test_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'y_test.parquet')
        log_print(f'Persisting y_test to \'{y_test_filename}\'...')
        self.y_test.to_frame().to_parquet(path=y_test_filename, index=False)
        # Metadata
        metadata_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'metadata.json')
        log_print(f'Persisting metadata to \'{metadata_filename}\'...')
        with open(metadata_filename, 'w') as fp:
            json.dump(self.metadata, fp, default=str)

    @function_call_logger
    def pipeline(self, preload=False, prepare=False) -> Self:
        if preload:
            self.preload()
        else:
            if prepare:
                self.prepare()
            self.load()
            self.sanitize()
            self.set_dtypes()
            # self.filter()
            self.encode()
            self.sort_columns()
            self.shrink_dtypes()
            self.analyze()
            self.select_features()
            self.train_test_split()
            self.quantile_attempt()
            # self.resample_hardness()
            # self.resample_torch()
            self.update_metadata()
            self.save()
        return self
