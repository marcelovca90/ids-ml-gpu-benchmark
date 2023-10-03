import json
import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import psutil
import torch
import torchsampler
from fastai.tabular.all import df_shrink
from featurewiz import FeatureWiz
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import (EditedNearestNeighbours,
                                     InstanceHardnessThreshold,
                                     RandomUnderSampler, TomekLinks)
from pandas_dq import dq_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchsampler import ImbalancedDatasetSampler
from typing_extensions import Self

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.balancer import BatchSizeHeuristic, CustomDataset
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
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=self.seed)

    @function_call_logger
    def resample_hardness(self) -> None:

        log_print('Training data value counts before resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.metadata['target_mappings_reverse'][str(
                unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape before resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

        iht = InstanceHardnessThreshold(
            sampling_strategy='all', random_state=self.seed, n_jobs=-1)
        self.X_train, self.y_train = iht.fit_resample(
            self.X_train, self.y_train)

        log_print('Training data value counts after resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.metadata['target_mappings_reverse'][str(
                unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape after resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

    @function_call_logger
    def resample_torch(self) -> None:
        log_print('Training data value counts before resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.metadata['target_mappings_reverse'][str(
                unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape before resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

        # Convert X_train DataFrame and y_train Series to the CustomDataset
        custom_dataset = CustomDataset(self.X_train, self.y_train)

        # Create an instance of ImbalancedDatasetSampler
        sampler = ImbalancedDatasetSampler(custom_dataset)

        # Estimate the optimal batch size
        batch_size = BatchSizeHeuristic.estimate(
            memory_usage_pct=0.5, dataset=custom_dataset)

        # Use the sampler in DataLoader
        data_loader = DataLoader(
            custom_dataset, batch_size=batch_size, sampler=sampler)

        # Now, during training, the DataLoader will provide batches with balanced class distributions.
        # This will help the model train more effectively on imbalanced data.

    @function_call_logger
    def resample_test(self) -> None:
        log_print('Training data value counts before resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.metadata['target_mappings_reverse'][str(
                unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape before resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

        _, class_counts = np.unique(self.y_train, return_counts=True)
        minority_class_size = np.min(class_counts)
        n_neighbors = int(np.sqrt(minority_class_size))

        smote = SMOTENC(sampling_strategy='auto',
                        categorical_features=self.metadata['cat_cols_mask'],
                        k_neighbors=n_neighbors, n_jobs=-1,
                        random_state=self.seed)

        enn = EditedNearestNeighbours(
            n_neighbors=n_neighbors, kind_sel='all', n_jobs=-1)

        smote_enn = SMOTEENN(sampling_strategy='auto', enn=enn,
                             smote=smote, random_state=self.seed, n_jobs=-1)

        self.X_train, self.y_train = smote_enn.fit_resample(
            self.X_train, self.y_train)

        log_print('Training data value counts after resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.metadata['target_mappings_reverse'][str(
                unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape after resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

    @function_call_logger
    def resample(self) -> None:
        log_print('Training data value counts before resampling...')
        total_samples = len(self.y_train)
        unique_labels, label_counts = np.unique(
            self.y_train, return_counts=True)
        rel_freqs = label_counts / total_samples
        for i in range(len(unique_labels)):
            label_name = self.metadata['target_encoding_reverse'][str(
                unique_labels[i])]
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
            label_name = self.metadata['target_encoding_reverse'][str(
                unique_labels[i])]
            log_print(
                f'({str(unique_labels[i]).rjust(2)}) {label_name.rjust(32)}' +
                f'\t{str(label_counts[i]).rjust(8)}\t{rel_freqs[i]:.3f}')
        log_print(
            f'Training data shape after resampling: ' +
            f'X_train={self.X_train.shape}, y_train={self.y_train.shape}')

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
    def quantile_attempt(self) -> None:
        print('quantile_attempt')
        pass

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
