import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from dtype_diet import optimize_dtypes, report_on_dataframe
from typing_extensions import Dict, List, Self, Union

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.stats import (log_data_types, log_memory_usage,
                                         log_value_counts)


class BasePreprocessingPipeline(ABC):

    def __init__(self, binarize=False) -> None:
        self.data: Union[str, Dict[str, pd.DataFrame]] = None
        self.folder: Union[str, Dict[str, str]] = None
        self.name: Union[str, List[str]] = None
        self.target: Union[str, Dict[str, str]] = None
        self.metadata: dict = dict()
        self.seed = 42
        self.binarize = binarize
        self.kind = 'Binary' if self.binarize else 'Multiclass'
        self.multiple = False

    @function_call_logger
    def preload(self) -> None:
        base_path = os.path.join(os.getcwd(), self.folder, 'generated')
        log_print(f'Loading cached files from \'{base_path}\'...')
        self.data = pd.read_parquet(
            os.path.join(base_path, self.name + '.parquet'))
        with open(os.path.join(base_path, 'metadata.json')) as json_file:
            self.metadata[self.name] = json.load(json_file)

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    # @function_call_logger
    # def sanitize(self) -> None:
    #     log_print('Value counts before sanitization:')
    #     log_value_counts(self.data, self.target)
    #     for col in self.data.columns.tolist():
    #         if 'int' in str(self.data[col].dtype):
    #             self.data[col].fillna(0, inplace=True)
    #         elif 'float' in str(self.data[col].dtype):
    #             self.data[col].fillna(0.0, inplace=True)
    #         elif 'object' == str(self.data[col].dtype):
    #             self.data[col].fillna('0', inplace=True)
    #     log_print('Value counts after sanitization:')
    #     log_value_counts(self.data, self.target)

    @function_call_logger
    def sanitize(self) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        tmp_target = {self.name: self.target} if not self.multiple else self.target
        for name, data in tmp_data.items():
            log_print('Value counts before sanitization:')
            log_value_counts(data, tmp_target[name])
            for col in data.columns:
                dtype = data[col].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    data[col] = data[col].fillna(0)
                elif pd.api.types.is_float_dtype(dtype):
                    data[col] = data[col].fillna(0.0)
                elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                    data[col] = data[col].fillna('0')
                else:
                    log_print(f"Skipping column {col} with unsupported dtype: {dtype}")
            log_print('Value counts after sanitization:')
            log_value_counts(data, tmp_target[name])

    @function_call_logger
    def set_dtypes(self) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        for _, data in tmp_data.items():
            log_print(f"Data types and memory usage before conversion:")
            log_data_types(data)
            log_memory_usage(data)
            for col in data.columns.tolist():
                try:
                    data[col] = pd.to_numeric(data[col], errors='raise')
                except Exception as e:
                    pass
            # data[self.target] = data[self.target].astype('category')
            log_print(f"Data types and memory usage before conversion:")
            log_data_types(data)
            log_memory_usage(data)

    def remove_na_duplicates(self) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        for _, data in tmp_data.items():
            num_duplicates_before = data.duplicated().sum()
            num_nas_before = data.isna().sum().sum()
            log_print(f"Duplicates before cleaning: {num_duplicates_before}")
            log_print(f"NAs before cleaning: {num_nas_before}")
            data.drop_duplicates(inplace=True)
            data.dropna(inplace=True)
            num_duplicates_after = data.duplicated().sum()
            num_nas_after = data.isna().sum().sum()
            log_print(f"Duplicates after cleaning: {num_duplicates_after}")
            log_print(f"NAs after cleaning: {num_nas_after}")
            log_print(f"Memory usage after cleaning:")
            log_memory_usage(data)

    @function_call_logger
    def shrink_dtypes(self) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        for _, data in tmp_data.items():
            log_print(f"Data types and memory usage before shrinkage:")
            log_data_types(data)
            log_memory_usage(data)
            df_report = report_on_dataframe(data, unit="MB", optimize="computation")
            data = optimize_dtypes(data, df_report)
            for col in data.select_dtypes(include=['integer', 'float']).columns:
                data[col] = pd.to_numeric(data[col], downcast='integer' if data[col].dtype.kind == 'i' else 'float')
            log_print(f"Data types and memory usage after shrinkage:")
            log_data_types(data)
            log_memory_usage(data)

    @function_call_logger
    def sort_columns(self) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        tmp_target = {self.name: self.target} if not self.multiple else self.target
        for name, data in tmp_data.items():
            cols = [x for x in data.columns.tolist() if x != tmp_target[name]]
            cols.extend([tmp_target[name]])
            log_print(f'Columns sorted according to {cols}.')
            data = data.reindex(columns=cols)

    @function_call_logger
    def reset_index(self) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        for _, data in tmp_data.items():
            data.reset_index(drop=True, inplace=True)

    @function_call_logger
    def update_metadata(self) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        tmp_target = {self.name: self.target} if not self.multiple else self.target
        for name, data in tmp_data.items():
            self.metadata[name] = {
                'multiple': self.multiple,
                'kind': self.kind,
                'columns': data.columns.tolist(),
                'description': data.describe().to_dict(),
                'dtypes': data.dtypes.to_dict(),
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'value_counts': data[tmp_target[name]].value_counts().to_dict()
            }

    @function_call_logger
    def save(self, csv=False, parquet=True, metadata=True) -> None:
        tmp_data = {self.name: self.data} if not self.multiple else self.data
        tmp_folder = {self.name: self.folder} if not self.multiple else self.folder
        tmp_metadata = {self.name: self.metadata} if not self.multiple else self.metadata
        # Dataset (CSV)
        if csv:
            for name, data in tmp_data.items():
                kind_suffix = f'_{self.kind}' \
                    if f'_{self.kind}' not in name else ''
                csv_filename = os.path.join(
                    os.getcwd(), tmp_folder[name], 'generated',
                    f'{name}{kind_suffix}.csv'
                )
                os.makedirs(Path(csv_filename).parent, exist_ok=True)
                log_print(f'Persisting dataset to \'{csv_filename}\'...')
                data.to_csv(path_or_buf=csv_filename, header=True, index=False)
        # Dataset (Parquet)
        if parquet:
            for name, data in tmp_data.items():
                kind_suffix = f'_{self.kind}' \
                    if f'_{self.kind}' not in name else ''
                parquet_filename = os.path.join(
                    os.getcwd(), tmp_folder[name], 'generated',
                    f'{name}{kind_suffix}.parquet'
                )
                log_print(f'Persisting dataset to \'{parquet_filename}\'...')
                data.to_parquet(path=parquet_filename, index=False)
        # Metadata
        if metadata:
            for name, data in tmp_data.items():
                kind_suffix = f'_{self.kind}' \
                    if f'_{self.kind}' not in name else ''
                metadata_filename = os.path.join(
                    os.getcwd(), tmp_folder[name], 'generated',
                    f'{name}{kind_suffix}.json'
                )
                log_print(f'Persisting metadata to \'{metadata_filename}\'...')
                with open(metadata_filename, 'w') as fp:
                    json.dump(tmp_metadata[name], fp, default=str, indent=4)

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
