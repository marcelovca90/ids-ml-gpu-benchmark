import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.utils import _replace_values

sys.path.append(Path(__file__).absolute().parent.parent)

class MQTT_IoT_IDS2020_BiflowFeatures(BasePreprocessingPipeline):

    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'mqtt_iot_ids2020')
        self.name = 'MQTT_IoT_IDS2020_BiflowFeatures'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'biflow_features')

        def filter_fn(x): return x.endswith('.csv')
        csv_files = list(filter(filter_fn, os.listdir(work_folder)))

        for base_filename in csv_files:
            filename_csv = os.path.join(work_folder, base_filename)
            filename_parquet = filename_csv.replace('.csv', '.parquet')
            with open(filename_csv, "r+") as f:
                content = f.readlines()
                f.truncate(0)
                f.seek(0)
                f.write(content[0])
                for line in content[1:]:
                    if not line.startswith('timestamp'):
                        f.write(line)
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df = pd.read_csv(filepath_or_buffer=filename_csv,
                             header=0, nrows=None, low_memory=False)
            df = df.drop(columns=['ip_src', 'ip_dst'])
            df = df.rename(columns={'is_attack': self.target})
            _replace_values(df, self.target,   0, 'normal')
            _replace_values(df, self.target, '0', 'normal')
            _replace_values(df, self.target,   1, base_filename.replace(
                'biflow_', '').replace('.csv', ''))
            _replace_values(df, self.target, '1', base_filename.replace(
                'biflow_', '').replace('.csv', ''))
            if self.binarize:
                df[self.target] = np.where(
                    (df[self.target] == 'normal'), 'Benign', 'Malign'
                )
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
            log_print(f'Loading parquet file \'{base_filename}\'.')
            full_filename = os.path.join(work_folder, base_filename)
            df = pd.read_parquet(full_filename)
            data_frames.append(df)
            log_print(f'Loaded parquet file \'{base_filename}\'.')
        self.data = pd.concat(data_frames, copy=False)
