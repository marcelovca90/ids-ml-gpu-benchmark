

import json
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from fastai.tabular.all import df_shrink
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.selection.drop_constant_features import \
    DropConstantFeatures
from feature_engine.selection.drop_correlated_features import \
    DropCorrelatedFeatures
from feature_engine.selection.drop_duplicate_features import \
    DropDuplicateFeatures
from feature_engine.transformation import YeoJohnsonTransformer
from pandas_dq import dq_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

    @function_call_logger
    def preload(self) -> None:
        base_path = os.path.join(os.getcwd(), self.folder, 'generated')
        log_print(f'Loading cached files from \'{base_path}\'...')
        self.data = pd.read_parquet(
            os.path.join(base_path, self.name + '.parquet'))
        with open(os.path.join(base_path, 'metadata.json')) as json_file:
            self.metadata = json.load(json_file)

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
        num_duplicates_before = self.data.duplicated().sum()
        num_nas_before = self.data.isna().sum().sum()
        log_print(f"Duplicates before cleaning: {num_duplicates_before}")
        log_print(f"NAs before cleaning: {num_nas_before}")
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        num_duplicates_after = self.data.duplicated().sum()
        num_nas_after = self.data.isna().sum().sum()
        log_print(f"Duplicates after cleaning: {num_duplicates_after}")
        log_print(f"NAs after cleaning: {num_nas_after}")

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        for col in self.data.columns:
            if 'int' in str(self.data[col].dtype):
                self.data[col].fillna(0, inplace=True)
            elif 'float' in str(self.data[col].dtype):
                self.data[col].fillna(0.0, inplace=True)
            elif 'object' == str(self.data[col].dtype):
                self.data[col].fillna('0', inplace=True)
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)

    @function_call_logger
    def set_dtypes(self) -> None:
        log_print(f"Data types and memory usage before conversion:")
        log_data_types(self.data)
        log_memory_usage(self.data)
        for col in self.data.columns:
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='raise')
            except Exception as e:
                pass
        self.data[self.target] = self.data[self.target].astype('category')
        log_print(f"Data types and memory usage before conversion:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def drop_irrelevant_features(self) -> None:

        feature_cols = [x for x in self.data.columns if x != self.target]

        constant_filter = DropConstantFeatures(variables=feature_cols)
        filtered_data = constant_filter.fit_transform(
            self.data.drop(columns=[self.target]), None)
        self.data = pd.concat([filtered_data, self.data[self.target]], axis=1)
        dropped_cols = constant_filter.features_to_drop_
        feature_cols = [x for x in feature_cols if x not in dropped_cols]
        log_print(f'Constant columns {dropped_cols} were dropped.')

        duplicate_filter = DropDuplicateFeatures(variables=feature_cols)
        filtered_data = duplicate_filter.fit_transform(
            self.data.drop(columns=[self.target]), None)
        self.data = pd.concat([filtered_data, self.data[self.target]], axis=1)
        dropped_cols = duplicate_filter.features_to_drop_
        feature_cols = [x for x in feature_cols if x not in dropped_cols]
        log_print(f'Duplicate columns {dropped_cols} were dropped.')

        correlated_filter = DropCorrelatedFeatures(
            variables=feature_cols, threshold=1.0)
        filtered_data = correlated_filter.fit_transform(
            self.data.drop(columns=[self.target]), None)
        self.data = pd.concat([filtered_data, self.data[self.target]], axis=1)
        dropped_cols = correlated_filter.features_to_drop_
        feature_cols = [x for x in feature_cols if x not in dropped_cols]
        log_print(f'Correlated columns {dropped_cols} were dropped.')

    @function_call_logger
    def encode(self) -> None:

        ordinal, one_hot, yeo_johnson = [], [], []

        n_uniques = {c: self.data[c].nunique() for c in self.data.columns}

        for col in self.data.drop(columns=[self.target]).columns.tolist():
            if pd.api.types.is_numeric_dtype(self.data.dtypes[col]):
                if n_uniques[col] > 1:
                    diffs = np.diff(self.data[col].values)
                    is_monotonic = all(diffs >= 0) or all(diffs <= 0)
                    if is_monotonic:
                        ordinal.append(col)
                    elif n_uniques[col] <= 10:
                        one_hot.append(col)
                    else:
                        yeo_johnson.append(col)
            elif pd.api.types.is_string_dtype(self.data.dtypes[col]):
                if n_uniques[col] > 1 and n_uniques[col] <= 10:
                    one_hot.append(col)

        categorical_mappings = {x: 'category' for x in one_hot}
        self.data = self.data.astype(categorical_mappings)

        if ordinal:
            ord_enc = OrdinalEncoder(variables=ordinal)
            self.data = ord_enc.fit_transform(self.data, None)
            log_print(f"Applied OrdinalEncoder ({ord_enc.encoder_dict_})")

        if one_hot:
            one_hot_enc = OneHotEncoder(variables=one_hot)
            self.data = one_hot_enc.fit_transform(self.data, None)
            log_print(f"Applied OneHotEncoder ({one_hot_enc.encoder_dict_})")

        if yeo_johnson:
            yjt = YeoJohnsonTransformer(variables=yeo_johnson)
            self.data = yjt.fit_transform(self.data, None)
            log_print(f"Applied YeoJohnsonTransformer ({yjt.lambda_dict_})")

        lbl_enc = LabelEncoder()
        self.data[self.target] = lbl_enc.fit_transform(self.data[self.target])
        log_print(f"Applied LabelEncoder ({self.target} @ {lbl_enc.classes_})")

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
    def update_metadata(self) -> None:
        self.metadata['columns'] = self.data.columns.to_list()
        self.metadata['description'] = self.data.describe().to_dict()
        self.metadata['dtypes'] = self.data.dtypes.to_dict()
        self.metadata['shape'] = self.data.shape
        self.metadata['memory_usage'] = self.data.memory_usage(deep=True).sum()
        self.metadata['value_counts'] = \
            self.data[self.target].value_counts().to_dict()

    @function_call_logger
    def save(self) -> None:
        # Dataset (CSV)
        csv_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', self.name + '.csv')
        log_print(f'Persisting dataset to \'{csv_filename}\'...')
        self.data.to_csv(path_or_buf=csv_filename, header=True, index=False)
        # Dataset (Parquet)
        parquet_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', self.name + '.parquet')
        log_print(f'Persisting dataset to \'{parquet_filename}\'...')
        self.data.to_parquet(path=parquet_filename, index=False)
        # Metadata
        metadata_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'metadata.json')
        log_print(f'Persisting metadata to \'{metadata_filename}\'...')
        with open(metadata_filename, 'w') as fp:
            json.dump(self.metadata, fp, default=str)

    @function_call_logger
    def pipeline(self, preload=False) -> Self:
        if preload:
            self.preload()
        else:
            self.prepare()
            self.load()
            self.set_dtypes()
            self.sanitize()
            self.encode()
            self.shrink_dtypes()
            self.drop_irrelevant_features()
            self.remove_na_duplicates()
            self.sort_columns()
            self.reset_index()
            self.update_metadata()
            self.save()
        return self
