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


class IoT_23(BasePreprocessingPipeline):

    def __init__(self) -> None:
        super().__init__()
        self.folder = os.path.join('datasets', 'iot_23')
        self.name = 'IoT_23'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        def filter_fn(x): return ('bro' in x)
        all_folders = [x[0] for x in os.walk(work_folder)]
        sub_folders = list(filter(filter_fn, all_folders))
        base_filename = 'conn.log.labeled'

        for folder in sub_folders:

            csv_filename = os.path.join(folder, base_filename)
            parquet_filename = csv_filename.replace('.labeled', '.parquet')
            col_names = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h',
                         'id.resp_p', 'proto', 'service', 'duration',
                         'orig_bytes', 'resp_bytes', 'conn_state',
                         'local_orig', 'local_resp', 'missed_bytes', 'history',
                         'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
                         'resp_ip_bytes', 'label']
            col_drops = ['ts', 'uid', 'service', 'local_orig', 'local_resp',
                         'history', 'id.orig_h', 'id.resp_h']
            log_print(f'Started converting file in \'{folder}\' to parquet.')

            # Get the total number of lines in the CSV file for progress bar
            chunk_size = _determine_best_chunksize(csv_filename, '\t', '#')
            total_lines = sum(1 for _ in open(csv_filename))
            num_iters = total_lines // chunk_size

            # Define the CSV reader with the specified chunksize and comment
            csv_reader = pd.read_csv(csv_filename, sep='\t',  comment='#',
                                     chunksize=chunk_size, dtype=str,
                                     na_values='', names=col_names)

            # Create a Parquet writer using the first chunk to set the schema
            first_chunk = next(csv_reader).drop(columns=col_drops)
            table = pa.Table.from_pandas(first_chunk)
            with pq.ParquetWriter(parquet_filename, table.schema) as writer:
                writer.write_table(table)
                # Process each chunk of data and append to the Parquet file
                for chunk in tqdm(csv_reader, total=num_iters, unit='chunk'):
                    # Convert chunk to Arrow table
                    table = pa.Table.from_pandas(chunk.drop(
                        columns=col_drops), schema=table.schema)
                    writer.write_table(table)

            log_print(f'Finished converting file in \'{folder}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        def filter_fn(x): return ('bro' in x)
        all_folders = [x[0] for x in os.walk(work_folder)]
        sub_folders = filter(filter_fn, all_folders)
        base_filename = 'conn.log.parquet'
        data_frames = []
        for folder in sub_folders:
            full_filename = os.path.join(folder, base_filename)
            log_print(f'Started loading parquet files in \'{folder}\'.')
            df = pd.read_parquet(full_filename, engine='pyarrow')
            df = df.drop_duplicates()
            df = df.dropna()
            data_frames.append(df)
            log_print(f'Finished loading parquet files in \'{folder}\'.')
        self.data = pd.concat(data_frames, copy=False)

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        _replace_values(self.data, 'duration',   '-',                                                           np.float64(0.0))                            # noqa
        _replace_values(self.data, 'orig_bytes', '-',                                                           np.uint64(0))                               # noqa
        _replace_values(self.data, 'resp_bytes', '-',                                                           np.uint64(0))                               # noqa
        _replace_values(self.data, self.target,      '-   Malicious   Attack',                                  'Attack')                                   # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   Attack',                            'Attack')                                   # noqa
        _replace_values(self.data, self.target,      '(empty)   Benign   -',                                    'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '-   benign   -',                                          'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '-   Benign   -',                                          'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '(empty)   Benign   -',                                    'Benign')                                   # noqa
        _replace_values(self.data, self.target,      'CARhxZ3hLNVO3xYFok   Benign   -',                         'Benign')                                   # noqa
        _replace_values(self.data, self.target,      'COLnd035cNITygYHp3   Benign   -',                         'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C',                                     'C&C')                                      # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   C&C',                               'C&C')                                      # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-FileDownload',                        'C&C-FileDownload')                         # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat',                           'C&C-HeartBeat')                            # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   C&C-HeartBeat',                     'C&C-HeartBeat')                            # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat-Attack',                    'C&C-HeartBeat-Attack')                     # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat-FileDownload',              'C&C-HeartBeat-FileDownload')               # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat-PartOfAHorizontalPortScan', 'C&C-HeartBeat-PartOfAHorizontalPortScan')  # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-Mirai',                               'C&C-Mirai')                                # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-PartOfAHorizontalPortScan',           'C&C-PartOfAHorizontalPortScan')            # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-Torii',                               'C&C-Torii')                                # noqa
        _replace_values(self.data, self.target,      '-   Malicious   DDoS',                                    'DDoS')                                     # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   DDoS',                              'DDoS')                                     # noqa
        _replace_values(self.data, self.target,      '-   Malicious   FileDownload',                            'FileDownload')                             # noqa
        _replace_values(self.data, self.target,      '-   Malicious   Okiru',                                   'Okiru')                                    # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   Okiru',                             'Okiru')                                    # noqa
        _replace_values(self.data, self.target,      '-   Malicious   Okiru-Attack',                            'Okiru-Attack')                             # noqa
        _replace_values(self.data, self.target,      '-   Malicious   PartOfAHorizontalPortScan',               'PartOfAHorizontalPortScan')                # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   PartOfAHorizontalPortScan',         'PartOfAHorizontalPortScan')                # noqa
        _replace_values(self.data, self.target,      '-   Malicious   PartOfAHorizontalPortScan-Attack',        'PartOfAHorizontalPortScan-Attack')         # noqa
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)

    @function_call_logger
    def set_dtypes(self) -> None:
        log_print('Data types before manual definition:')
        log_data_types(self.data)
        self.data = self.data.infer_objects().astype({
            'id.orig_p': np.uint64,
            'id.resp_p': np.uint64,
            'duration': np.float64,
            'orig_bytes': np.uint64,
            'resp_bytes': np.uint64,
            'missed_bytes': np.uint64,
            'orig_pkts': np.uint64,
            'orig_ip_bytes': np.uint64,
            'resp_pkts': np.uint64,
            'resp_ip_bytes': np.uint64,
            self.target: 'category'
        })
        log_print('Data types after manual definition:')
        log_data_types(self.data)

    @function_call_logger
    def encode(self) -> None:
        self.data = _one_hot_encode(self.data, 'proto')
        self.data, _ = _label_encode(self.data, 'conn_state')
        self.data, target_mappings = _label_encode(self.data, self.target)
        features = self.data.columns[self.data.columns != self.target]
        cat_cols = [x for x in features if 'proto' in x or 'conn_state' in x]
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
