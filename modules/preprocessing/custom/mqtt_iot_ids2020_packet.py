import os

import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts
from modules.preprocessing.utils import (_label_encode, _one_hot_encode,
                                         _replace_values)


class MQTT_IoT_IDS2020_PacketFeatures(BasePreprocessingPipeline):

    def __init__(self) -> None:
        super().__init__()
        self.folder = os.path.join('datasets', 'mqtt_iot_ids2020')
        self.name = 'MQTT_IoT_IDS2020'
        self.target = 'is_attack'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'packet_features')

        def filter_fn(x): return x.endswith('.csv')
        csv_files = list(filter(filter_fn, os.listdir(work_folder)))

        for base_filename in csv_files:
            filename_csv = os.path.join(work_folder, base_filename)
            filename_parquet = filename_csv.replace('.csv', '.parquet')
            df = pd.read_csv(filepath_or_buffer=filename_csv,
                             header=0, nrows=None, low_memory=False)
            df = df.drop(columns=['timestamp', 'src_ip', 'dst_ip'])
            _replace_values(df, 'is_attack',   0, 'normal')
            _replace_values(df, 'is_attack', '0', 'normal')
            _replace_values(df, 'is_attack',   1, base_filename.replace(
                'packet_', '').replace('.csv', ''))
            _replace_values(df, 'is_attack', '1', base_filename.replace(
                'packet_', '').replace('.csv', ''))
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df.to_parquet(filename_parquet)
            log_print(f'Converted file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'packet_features')

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
