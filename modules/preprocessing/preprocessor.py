import json
import os
from abc import ABC, abstractmethod

import pandas as pd
from fastai.tabular.all import df_shrink
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

    @abstractmethod
    def load(self) -> None:
        pass

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        for col in self.data.columns.tolist():
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
        for col in self.data.columns.tolist():
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='raise')
            except Exception as e:
                pass
        # self.data[self.target] = self.data[self.target].astype('category')
        log_print(f"Data types and memory usage before conversion:")
        log_data_types(self.data)
        log_memory_usage(self.data)

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
        log_print(f"Memory usage after cleaning:")
        log_memory_usage(self.data)

    @function_call_logger
    def shrink_dtypes(self) -> None:
        log_print(f"Data types and memory usage before shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)
        self.data = df_shrink(self.data, obj2cat=False, int2uint=False)
        log_print(f"Data types and memory usage after shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def sort_columns(self) -> None:
        cols = [x for x in self.data.columns.tolist() if x != self.target]
        cols.extend([self.target])
        log_print(f'Columns sorted according to {cols}.')
        self.data = self.data.reindex(columns=cols)

    @function_call_logger
    def reset_index(self) -> None:
        self.data.reset_index(drop=True, inplace=True)

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
            os.getcwd(), self.folder, 'generated', self.name + '.json')
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
            self.sanitize()
            self.set_dtypes()
            self.remove_na_duplicates()
            self.shrink_dtypes()
            self.sort_columns()
            self.reset_index()
            self.update_metadata()
            self.save()
        return self
