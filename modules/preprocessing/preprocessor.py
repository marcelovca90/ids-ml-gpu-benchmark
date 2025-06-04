import json
import os
from abc import ABC, abstractmethod

import pandas as pd
from dtype_diet import optimize_dtypes, report_on_dataframe
from typing_extensions import Self

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.stats import (
    log_data_types, log_memory_usage, log_value_counts
)


class BasePreprocessingPipeline(ABC):

    def __init__(self, binarize=False) -> None:
        self.data: pd.DataFrame = None
        self.folder: str = None
        self.name: str = None
        self.target: str = None
        self.metadata: dict = {}
        self.complexity: dict = {}
        self.seed = 42
        self.binarize = binarize
        self.kind = 'Binary' if self.binarize else 'Multiclass'

    @function_call_logger
    def preload(self) -> None:
        base_path = os.path.join(os.getcwd(), self.folder, 'generated')
        log_print(f'Loading cached files from \'{base_path}\'...')
        self.data = pd.read_parquet(os.path.join(base_path, self.name + '.parquet'))
        with open(os.path.join(base_path, 'metadata.json')) as json_file:
            self.metadata[self.name] = json.load(json_file)

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
        for col in self.data.columns:
            dtype = self.data[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                self.data[col] = self.data[col].fillna(0)
            elif pd.api.types.is_float_dtype(dtype):
                self.data[col] = self.data[col].fillna(0.0)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                self.data[col] = self.data[col].fillna('0')
            else:
                log_print(f"Skipping column {col} with unsupported dtype: {dtype}")
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)

    @function_call_logger
    def round(self, decimals=3) -> None:
        log_print(f"Number of unique values per column before rounding to {decimals} decimal places:")
        for col in self.data.columns:
            log_print(f"{col}\t{self.data[col].nunique()}")
        self.data = self.data.round(decimals=decimals)
        log_print(f"Number of unique values per column after rounding to {decimals} decimal places:")
        for col in self.data.columns:
            log_print(f"{col}\t{self.data[col].nunique()}")

    @function_call_logger
    def set_dtypes(self) -> None:
        log_print("Data types and memory usage before conversion:")
        log_data_types(self.data)
        log_memory_usage(self.data)
        for col in self.data.columns:
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='raise')
            except Exception:
                pass
        log_print("Data types and memory usage after conversion:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def remove_na_duplicates(self) -> None:
        num_nas_before = self.data.isna().sum().sum()
        num_duplicates_before = self.data.duplicated().sum()
        log_print(f"NAs before cleaning: {num_nas_before}")
        log_print(f"Duplicates before cleaning: {num_duplicates_before}")
        self.data.dropna(axis='columns', how='all', inplace=True)
        self.data.dropna(axis='index', how='any', inplace=True)
        self.data.drop_duplicates(inplace=True)
        num_nas_after = self.data.isna().sum().sum()
        num_duplicates_after = self.data.duplicated().sum()
        log_print(f"NAs after cleaning: {num_nas_after}")
        log_print(f"Duplicates after cleaning: {num_duplicates_after}")
        log_print(f"Memory usage after cleaning:")
        log_memory_usage(self.data)

    @function_call_logger
    def shrink_dtypes(self) -> None:
        log_print("Data types and memory usage before shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)
        df_report = report_on_dataframe(self.data, unit="MB", optimize="computation")
        self.data = optimize_dtypes(self.data, df_report)
        for col in self.data.select_dtypes(include=['integer', 'float']).columns:
            self.data[col] = pd.to_numeric(
                self.data[col], downcast='integer' if self.data[col].dtype.kind == 'i' else 'float')
        log_print("Data types and memory usage after shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def sort_columns(self) -> None:
        cols = [x for x in self.data.columns.tolist() if x != self.target]
        cols.append(self.target)
        log_print(f'Columns sorted according to {cols}.')
        self.data = self.data.reindex(columns=cols)

    @function_call_logger
    def reset_index(self) -> None:
        self.data.reset_index(drop=True, inplace=True)

    @function_call_logger
    def update_metadata(self) -> None:
        self.metadata[self.name] = {
            'kind': self.kind,
            'columns': self.data.columns.tolist(),
            'description': self.data.describe().to_dict(),
            'dtypes': self.data.dtypes.apply(str).to_dict(),
            'shape': self.data.shape,
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'value_counts': self.data[self.target].value_counts().to_dict(),
            'complexity': self.complexity.get(self.name, {})
        }

    @function_call_logger
    def compute_complexity(self, complexity_mode=None) -> None:
        if complexity_mode:
            if complexity_mode == 'cpu':
                from modules.preprocessing.complexity_cpu import (
                    compute_all_complexity_measures, smart_categorical_encode)
            elif complexity_mode == 'gpu':
                from modules.preprocessing.complexity_gpu import (
                    compute_all_complexity_measures, smart_categorical_encode)
            X, y = self.data.drop(columns=[self.target]), self.data[self.target]
            X, y = smart_categorical_encode(X, y)
            self.complexity[self.name] = compute_all_complexity_measures(X, y)
        else:
            self.complexity[self.name] = {}

    @function_call_logger
    def save(self, csv=False, parquet=True, metadata=True) -> None:
        kind_suffix = f'_{self.kind}' if f'_{self.kind}' not in self.name else ''
        base_dir = os.path.join(os.getcwd(), self.folder, 'generated')
        os.makedirs(base_dir, exist_ok=True)

        if csv:
            csv_filename = os.path.join(base_dir, f'{self.name}{kind_suffix}.csv')
            log_print(f'Persisting dataset to \'{csv_filename}\'...')
            self.data.to_csv(csv_filename, index=False)

        if parquet:
            parquet_filename = os.path.join(base_dir, f'{self.name}{kind_suffix}.parquet')
            log_print(f'Persisting dataset to \'{parquet_filename}\'...')
            self.data.to_parquet(parquet_filename, index=False)

        if metadata:
            metadata_filename = os.path.join(base_dir, f'{self.name}{kind_suffix}.json')
            log_print(f'Persisting metadata to \'{metadata_filename}\'...')
            with open(metadata_filename, 'w') as fp:
                json.dump(self.metadata[self.name], fp, indent=4, default=str)

    @function_call_logger
    def pipeline(self, preload=False, complexity_mode=None) -> Self:
        if preload:
            self.preload()
        else:
            self.prepare()
            self.load()
            self.sanitize()
            self.set_dtypes()
            self.round()
            self.remove_na_duplicates()
            self.shrink_dtypes()
            self.sort_columns()
            self.reset_index()
            self.compute_complexity(complexity_mode)
            self.update_metadata()
            self.save()
        return self
