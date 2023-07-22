import os
from abc import ABC, abstractmethod

import numpy as np
from fastai.tabular.all import df_shrink
from imblearn.under_sampling import TomekLinks
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from typing_extensions import Self

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.stats import (log_data_types, log_memory_usage,
                                         log_value_counts)


class BasePreprocessingPipeline(ABC):

    def __init__(self) -> None:
        self.data: DataFrame = None
        self.folder: str = None
        self.name: str = None
        self.target: str = None
        self.mappings: dict = None
        self.seed = 42
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    @abstractmethod
    def encode(self) -> None:
        pass

    @function_call_logger
    def filter(self) -> None:
        log_print(f"Value counts before filtering by frequency:")
        log_value_counts(self.data, self.target)
        vcd = self.data[self.target].value_counts(normalize=True).to_dict()
        kept_labels = [key for key, val in vcd.items() if val > 0.01/100.0]
        log_print(f'Dropping rows with frequency inferior to {0.01:.3f}% ...')
        filtered_labels = self.data[self.target].value_counts(
        ).index.drop(kept_labels)
        for label in filtered_labels:
            self.data = self.data.drop(
                self.data[self.data[self.target] == label].index)
        log_print(f"Value counts after filtering by frequency:")
        log_value_counts(self.data, self.target)

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def sanitize(self) -> None:
        pass

    @function_call_logger
    def save(self) -> None:
        csv_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', self.name + '.csv')
        log_print(f'Persisting CSV to \'{csv_filename}\'...')
        self.data.to_csv(path_or_buf=csv_filename, header=True, index=False)
        parquet_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', self.name + '.parquet')
        log_print(f'Persisting Parquet to \'{parquet_filename}\'...')
        self.data.to_parquet(path=parquet_filename, index=False)

    @abstractmethod
    def set_dtypes(self) -> None:
        pass

    @function_call_logger
    def shrink_dtypes(self) -> None:
        log_print(f"Data types and memory usage before shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)
        self.data = df_shrink(self.data)
        log_print(f"Data types and memory usage after shrinkage:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def sort_columns(self) -> None:
        cols = [x for x in self.data.columns.values if x not in [self.target]]
        cols.extend([self.target])
        log_print(f'Columns sorted according to {cols}.')
        self.data = self.data.reindex(columns=cols)

    @function_call_logger
    def train_test_split(self) -> None:
        X, y = self.data.drop(columns=[self.target]), self.data[self.target]
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=self.seed)
        self.X_train = X_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()

    @function_call_logger
    def resample(self) -> None:
        log_print(
            f'Training data shape before resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')
        tomek = TomekLinks(sampling_strategy='auto')
        self.X_train, self.y_train = tomek.fit_resample(
            self.X_train, self.y_train)
        log_print(
            f'Training data shape after resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

    @function_call_logger
    def setup(self) -> Self:
        self.load()
        self.sanitize()
        self.set_dtypes()
        self.filter()
        self.encode()
        self.sort_columns()
        self.shrink_dtypes()
        self.train_test_split()
        self.resample()
        self.save()
        return self
