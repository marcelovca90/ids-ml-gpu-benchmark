import json
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.tabular.all import df_shrink
from featurewiz import FeatureWiz
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     TomekLinks)
from pandas import DataFrame
from sklearn.cluster import OPTICS, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
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
        self.reverse_mappings: dict = None
        self.seed = 42
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    @function_call_logger
    def preload(self) -> None:
        base_path = os.path.join(os.getcwd(), self.folder, 'generated')
        log_print(f'Loading cached files from \'{base_path}\'...')
        self.data = pd.read_parquet(
            os.path.join(base_path, self.name + '.parquet'))
        with open(os.path.join(base_path, 'mappings.json')) as json_file:
            self.mappings = json.load(json_file)
        self.reverse_mappings = dict((v, k) for k, v in self.mappings.items())
        self.X_train = np.load(os.path.join(base_path, 'X_train.npy'))
        self.y_train = np.load(os.path.join(base_path, 'y_train.npy'))
        self.X_test = np.load(os.path.join(base_path, 'X_test.npy'))
        self.y_test = np.load(os.path.join(base_path, 'y_test.npy'))

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def sanitize(self) -> None:
        pass

    @abstractmethod
    def set_dtypes(self) -> None:
        pass

    @function_call_logger
    def filter(self) -> None:
        log_print(f"Value counts before filtering by frequency:")
        log_value_counts(self.data, self.target)
        vcd = self.data[self.target].value_counts(normalize=True).to_dict()
        kept_labels = [key for key, val in vcd.items() if val > 0.01/100.0]
        log_print(f'Dropping labels with frequency inferior to 0.01% ...')
        filtered_labels = self.data[self.target].value_counts(
        ).index.drop(kept_labels)
        for label in filtered_labels:
            self.data = self.data.drop(
                self.data[self.data[self.target] == label].index)
        self.data.reset_index(drop=True, inplace=True)
        log_print(f"Value counts after filtering by frequency:")
        log_value_counts(self.data, self.target)

    @abstractmethod
    def encode(self) -> None:
        pass

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
        self.data = df_shrink(self.data)
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
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=self.seed)
        self.X_train = X_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()

    @function_call_logger
    def resample(self) -> None:
        log_print('Training data value counts before resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.reverse_mappings[str(unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape before resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

        # Calculate the desired minority class samples for rebalancing.
        desired_minority_ratio = 0.01  # 1% for the minority class
        desired_minority_samples = int(desired_minority_ratio * total_samples)

        # Calculate the sampling_strategy for RandomUnderSampler, avoiding an
        # excessive request for samples in the minority class.
        sampling_strategy = {}
        for label, count in zip(unique_labels, label_counts):
            if count <= desired_minority_samples:
                # If the class already has fewer samples than
                # desired_minority_samples, retain all samples in the class
                # (no undersampling).
                sampling_strategy[label] = count
            else:
                # Else, undersample the class to the desired number of samples.
                sampling_strategy[label] = desired_minority_samples

        ru_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=self.seed)
        self.X_train, self.y_train = ru_sampler.fit_resample(
            self.X_train, self.y_train)

        tl_sampler = TomekLinks(sampling_strategy='auto', n_jobs=-1)
        self.X_train, self.y_train = tl_sampler.fit_resample(
            self.X_train, self.y_train)

        log_print('Training data value counts after resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.reverse_mappings[str(unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape after resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

    @function_call_logger
    def save(self) -> None:
        # Dataset
        dataset_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', self.name + '.parquet')
        log_print(f'Persisting dataset to \'{dataset_filename}\'...')
        self.data.to_parquet(path=dataset_filename, index=False)
        # Train data
        X_train_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'X_train.npy')
        log_print(f'Persisting X_train to \'{X_train_filename}\'...')
        np.save(file=X_train_filename, arr=self.X_train)
        y_train_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'y_train.npy')
        log_print(f'Persisting y_train to \'{y_train_filename}\'...')
        np.save(file=y_train_filename, arr=self.y_train)
        # Test data
        X_test_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'X_test.npy')
        log_print(f'Persisting X_test to \'{X_test_filename}\'...')
        np.save(file=X_test_filename, arr=self.X_test)
        y_test_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'y_test.npy')
        log_print(f'Persisting y_test to \'{y_test_filename}\'...')
        np.save(file=y_test_filename, arr=self.y_test)
        # Mappings
        json_filename = os.path.join(
            os.getcwd(), self.folder, 'generated', 'mappings.json')
        log_print(f'Persisting mappings to \'{json_filename}\'...')
        with open(json_filename, 'w') as fp:
            json.dump(self.mappings, fp)
        log_print(f'Mappings persisted to \'{json_filename}\'.')

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
            self.filter()
            self.encode()
            self.sort_columns()
            self.shrink_dtypes()
            self.select_features()
            self.train_test_split()
            self.resample()
            self.save()
        return self
