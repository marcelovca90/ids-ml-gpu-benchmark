import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts
from modules.preprocessing.utils import _replace_values

sys.path.append(Path(__file__).absolute().parent.parent)

class UNSW_NB15(BasePreprocessingPipeline):

    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'unsw_nb15')
        self.name = 'UNSW_NB15'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        filename_cols = os.path.join(work_folder, 'NUSW-NB15_features.csv')
        columns = []
        with open(filename_cols) as file:
            for line in file.readlines():
                if re.match(r'^\d+', line):
                    columns.append(line.split(',')[1])

        def filter_fn(x): return re.match(r'UNSW-NB15_\d\.csv', x)
        csv_files = list(filter(filter_fn, os.listdir(work_folder)))
        for base_filename in csv_files:
            filename_csv = os.path.join(work_folder, base_filename)
            filename_parquet = filename_csv.replace('.csv', '.parquet')
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df = pd.read_csv(filename_csv, header=None,
                             names=columns, low_memory=False)
            df = df.drop(columns=['srcip', 'dstip', 'Label'])
            df = df.rename(columns={'attack_cat': self.target})
            df.to_parquet(filename_parquet)
            log_print(f'Converted  file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        def filter_fn(x): return x.endswith('.parquet')
        parquet_files = list(filter(filter_fn, os.listdir(work_folder)))
        data_frames = []
        for base_filename in parquet_files:
            log_print(f'Loading parquet file \'{base_filename}\'.')
            full_filename = os.path.join(work_folder, base_filename)
            data_frames.append(pd.read_parquet(full_filename))
            log_print(f'Loaded parquet file \'{base_filename}\'.')
        self.data = pd.concat(data_frames, copy=False)

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        data_obj = self.data.select_dtypes(['object'])
        self.data[data_obj.columns] = data_obj.apply(lambda x: x.str.strip())
        self.data['sport'] = self.data['sport'].fillna('-1')
        self.data['ct_flw_http_mthd'] = self.data['ct_flw_http_mthd'].fillna(-1.0)
        self.data['is_ftp_login'] = self.data['is_ftp_login'].fillna(-1.0)
        self.data['ct_ftp_cmd'] = self.data['ct_ftp_cmd'].fillna('-1')
        self.data[self.target] = self.data[self.target].fillna('Normal')
        if self.binarize:
            self.data[self.target] = np.where(
                (self.data[self.target] == 'Benign'), 'Benign', 'Malign'
            )
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)
