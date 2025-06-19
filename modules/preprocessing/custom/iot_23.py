import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_memory_usage, log_value_counts

sys.path.append(Path(__file__).absolute().parent.parent)

class IoT_23(BasePreprocessingPipeline):

    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'iot_23')
        self.name = 'IoT_23'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        all_folders = [x[0] for x in os.walk(work_folder)]
        sub_folders = [x for x in all_folders if 'bro' in x]
        base_filename = 'conn.log.labeled'
        for folder in sub_folders:
            filename_csv = os.path.join(folder, base_filename)
            filename_parquet = filename_csv.replace('.labeled', '.parquet')
            col_names = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h',
                         'id.resp_p', 'proto', 'service', 'duration',
                         'orig_bytes', 'resp_bytes', 'conn_state',
                         'local_orig', 'local_resp', 'missed_bytes', 'history',
                         'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
                         'resp_ip_bytes', 'label']
            col_drops = ['ts', 'uid', 'service', 'local_orig', 'local_resp',
                         'history', 'id.orig_h', 'id.resp_h']
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df = pd.read_csv(filename_csv, sep='\t',  comment='#',
                             dtype=str, na_values='', names=col_names)
            df = df.drop(columns=col_drops)
            log_print(f'Sanitizing, dropping NAs, and duplicates of file \'{filename_csv}\'.')
            num_nas_before = df.isna().sum().sum()
            num_duplicates_before = df.duplicated().sum()
            mem_before = df.memory_usage(deep=False).sum()
            df = self.sanitize_partial(df, self.target)
            df = self.drop_na_duplicates_partial(df)
            num_nas_after = df.isna().sum().sum()
            num_duplicates_after = df.duplicated().sum()
            mem_after = df.memory_usage(deep=False).sum()
            log_print(f"# N/As: {num_nas_before} -> {num_nas_after}")
            log_print(f"# DUPs: {num_duplicates_before} -> {num_duplicates_after}")
            log_print(f"MEMORY: {mem_before / (1024 ** 2):.2f} MB -> {mem_after / (1024 ** 2):.2f} MB")
            log_print(f'Sanitized,  dropped  NAs, and duplicates of file \'{filename_csv}\'.')
            df.to_parquet(filename_parquet)
            log_print(f'Converted  file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        all_folders = [x[0] for x in os.walk(work_folder)]
        sub_folders = [x for x in all_folders if 'bro' in x]
        base_filename = 'conn.log.parquet'
        data_frames = []
        for folder in sub_folders:
            full_filename = os.path.join(folder, base_filename)
            log_print(f'Started  loading parquet files in \'{folder}\'.')
            df = pd.read_parquet(full_filename)
            data_frames.append(df)
            log_print(f'Finished loading parquet files in \'{folder}\'.')
        self.data = pd.concat(data_frames, copy=False)

    @function_call_logger
    def sanitize(self) -> None:
        pass
        # log_print('Value counts before sanitization:')
        # log_value_counts(self.data, self.target)

        # # Create mappings per column
        # feature_replacements = {
        #     'duration'   : { '-' : '0.0' },
        #     'orig_bytes' : { '-' : '0'   },
        #     'resp_bytes' : { '-' : '0'   }
        # }
        # target_replacements = {
        #     '-   Malicious   Attack'                                 : 'Attack',
        #     '(empty)   Malicious   Attack'                           : 'Attack',
        #     '(empty)   Benign   -'                                   : 'Benign',
        #     '-   benign   -'                                         : 'Benign',
        #     '-   Benign   -'                                         : 'Benign',
        #     '(empty)   Benign   -'                                   : 'Benign',
        #     'CARhxZ3hLNVO3xYFok   Benign   -'                        : 'Benign',
        #     'COLnd035cNITygYHp3   Benign   -'                        : 'Benign',
        #     '-   Malicious   C&C'                                    : 'C&C',
        #     '(empty)   Malicious   C&C'                              : 'C&C',
        #     '-   Malicious   C&C-FileDownload'                       : 'C&C-FileDownload',
        #     '-   Malicious   C&C-HeartBeat'                          : 'C&C-HeartBeat',
        #     '(empty)   Malicious   C&C-HeartBeat'                    : 'C&C-HeartBeat',
        #     '-   Malicious   C&C-HeartBeat-Attack'                   : 'C&C-HeartBeat-Attack',
        #     '-   Malicious   C&C-HeartBeat-FileDownload'             : 'C&C-HeartBeat-FileDownload',
        #     '-   Malicious   C&C-HeartBeat-PartOfAHorizontalPortScan': 'C&C-HeartBeat-PartOfAHorizontalPortScan',
        #     '-   Malicious   C&C-Mirai'                              : 'C&C-Mirai',
        #     '-   Malicious   C&C-PartOfAHorizontalPortScan'          : 'C&C-PartOfAHorizontalPortScan',
        #     '-   Malicious   C&C-Torii'                              : 'C&C-Torii',
        #     '-   Malicious   DDoS'                                   : 'DDoS',
        #     '(empty)   Malicious   DDoS'                             : 'DDoS',
        #     '-   Malicious   FileDownload'                           : 'FileDownload',
        #     '-   Malicious   Okiru'                                  : 'Okiru',
        #     '(empty)   Malicious   Okiru'                            : 'Okiru',
        #     '-   Malicious   Okiru-Attack'                           : 'Okiru-Attack',
        #     '-   Malicious   PartOfAHorizontalPortScan'              : 'PartOfAHorizontalPortScan',
        #     '(empty)   Malicious   PartOfAHorizontalPortScan'        : 'PartOfAHorizontalPortScan',
        #     '-   Malicious   PartOfAHorizontalPortScan-Attack'       : 'PartOfAHorizontalPortScan-Attack',
        # }

        # # Apply all replacements efficiently
        # for column, mapping in feature_replacements.items():
        #     self.data[column] = pd.to_numeric(self.data[column].replace(mapping))
        # self.data[self.target] = self.data[self.target].replace(target_replacements)

        # if self.binarize:
        #     self.data[self.target] = np.where(
        #         (self.data[self.target] == 'Benign'), 'Benign', 'Malign'
        #     )
        # log_print('Value counts after sanitization:')
        # log_value_counts(self.data, self.target)

    @function_call_logger
    def sanitize_partial(self, data, target) -> pd.DataFrame:
        # Create mappings per column (features)
        feature_replacements = {
            'duration'   : { '-' : '0.0' },
            'orig_bytes' : { '-' : '0'   },
            'resp_bytes' : { '-' : '0'   }
        }
        # Create mappings per column (target)
        target_replacements = {
            '-   Malicious   Attack'                                 : 'Attack',
            '(empty)   Malicious   Attack'                           : 'Attack',
            '(empty)   Benign   -'                                   : 'Benign',
            '-   benign   -'                                         : 'Benign',
            '-   Benign   -'                                         : 'Benign',
            '(empty)   Benign   -'                                   : 'Benign',
            'CARhxZ3hLNVO3xYFok   Benign   -'                        : 'Benign',
            'COLnd035cNITygYHp3   Benign   -'                        : 'Benign',
            '-   Malicious   C&C'                                    : 'C&C',
            '(empty)   Malicious   C&C'                              : 'C&C',
            '-   Malicious   C&C-FileDownload'                       : 'C&C-FileDownload',
            '-   Malicious   C&C-HeartBeat'                          : 'C&C-HeartBeat',
            '(empty)   Malicious   C&C-HeartBeat'                    : 'C&C-HeartBeat',
            '-   Malicious   C&C-HeartBeat-Attack'                   : 'C&C-HeartBeat-Attack',
            '-   Malicious   C&C-HeartBeat-FileDownload'             : 'C&C-HeartBeat-FileDownload',
            '-   Malicious   C&C-HeartBeat-PartOfAHorizontalPortScan': 'C&C-HeartBeat-PartOfAHorizontalPortScan',
            '-   Malicious   C&C-Mirai'                              : 'C&C-Mirai',
            '-   Malicious   C&C-PartOfAHorizontalPortScan'          : 'C&C-PartOfAHorizontalPortScan',
            '-   Malicious   C&C-Torii'                              : 'C&C-Torii',
            '-   Malicious   DDoS'                                   : 'DDoS',
            '(empty)   Malicious   DDoS'                             : 'DDoS',
            '-   Malicious   FileDownload'                           : 'FileDownload',
            '-   Malicious   Okiru'                                  : 'Okiru',
            '(empty)   Malicious   Okiru'                            : 'Okiru',
            '-   Malicious   Okiru-Attack'                           : 'Okiru-Attack',
            '-   Malicious   PartOfAHorizontalPortScan'              : 'PartOfAHorizontalPortScan',
            '(empty)   Malicious   PartOfAHorizontalPortScan'        : 'PartOfAHorizontalPortScan',
            '-   Malicious   PartOfAHorizontalPortScan-Attack'       : 'PartOfAHorizontalPortScan-Attack',
        }
        # Apply all replacements efficiently
        for column, mapping in feature_replacements.items():
            data[column] = pd.to_numeric(data[column].replace(mapping))
        data[target] = data[target].replace(target_replacements)
        # Binarize if defined
        if self.binarize:
            data[target] = np.where(
                (data[target] == 'Benign'), 'Benign', 'Malign'
            )
        return data

    @function_call_logger
    def drop_na_duplicates_partial(self, data) -> pd.DataFrame:
        data.dropna(axis='index', how='any', inplace=True)
        data.drop_duplicates(inplace=True)
        return data