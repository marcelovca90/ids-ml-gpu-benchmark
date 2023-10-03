import os
import tempfile
from itertools import islice

import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
from tqdm import tqdm

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_data_types, log_value_counts
from modules.preprocessing.utils import (_determine_best_chunksize,
                                         _label_encode, _one_hot_encode,
                                         _replace_values)


class MQTT_IoT_IDS2020_BiflowFeatures(BasePreprocessingPipeline):

    def __init__(self) -> None:
        super().__init__()
        self.folder = os.path.join('datasets', 'mqtt_iot_ids2020')
        self.name = 'MQTT_IoT_IDS2020'
        self.target = 'is_attack'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'biflow_features')

        def filter_fn(x): return x.endswith('.csv')
        csv_files = list(filter(filter_fn, os.listdir(work_folder)))

        for base_filename in csv_files:
            filename_csv = os.path.join(work_folder, base_filename)
            filename_parquet = filename_csv.replace('.csv', '.parquet')
            df = pd.read_csv(filepath_or_buffer=filename_csv,
                             header=0, nrows=None, low_memory=False)
            df = df.drop(columns=['ip_src', 'ip_dst'])
            _replace_values(df, 'is_attack',   0, 'normal')
            _replace_values(df, 'is_attack', '0', 'normal')
            _replace_values(df, 'is_attack',   1, base_filename.replace(
                'biflow_', '').replace('.csv', ''))
            _replace_values(df, 'is_attack', '1', base_filename.replace(
                'biflow_', '').replace('.csv', ''))
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df.to_parquet(filename_parquet)
            log_print(f'Converted file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'biflow_features')

        def filter_fn(x): return x.endswith('.parquet')
        parquet_files = list(filter(filter_fn, os.listdir(work_folder)))

        data_frames = []
        for base_filename in parquet_files:
            log_print(f'Loading parquet files in \'{work_folder}\'.')
            full_filename = os.path.join(work_folder, base_filename)
            df = pd.read_parquet(full_filename, engine='pyarrow')
            df = df.drop_duplicates()
            df = df.dropna()
            data_frames.append(df)
            log_print(f'Loaded parquet files in \'{work_folder}\'.')
        self.data = pd.concat(data_frames, copy=False)

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        self.data.dropna(axis='index')
        self.data.drop_duplicates()
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)

    @function_call_logger
    def set_dtypes(self) -> None:
        log_print('Data types before inference:')
        log_data_types(self.data)

    @function_call_logger
    def encode(self) -> None:
        self.data = _one_hot_encode(self.data, 'proto')
        self.data, target_mappings = _label_encode(self.data, self.target)
        features = self.data.columns[self.data.columns != self.target]
        cat_cols = [x for x in features if 'proto' in x]
        cat_mask = [x in cat_cols for x in features]
        num_cols = [x for x in features if x not in cat_cols]
        num_mask = [x in num_cols for x in features]
        self.metadata['cat_cols'] = cat_cols
        self.metadata['cat_cols_mask'] = cat_mask
        self.metadata['num_cols'] = num_cols
        self.metadata['num_cols_mask'] = num_mask
        self.metadata['target_mappings'] = target_mappings
        self.metadata['target_mappings_reverse'] = \
            dict((v, k) for k, v in target_mappings.items())
