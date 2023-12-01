import os
import re

import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts
from modules.preprocessing.utils import _replace_values


class BoT_IoT_Micro(BasePreprocessingPipeline):

    def __init__(self) -> None:
        super().__init__()
        self.folder = os.path.join('datasets', 'bot_iot')
        self.name = 'BoT_IoT_Micro'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'Entire Dataset')

        filename_cols = os.path.join(
            work_folder, 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv')
        columns = []
        with open(filename_cols) as file:
            columns = [x.strip() for x in file.readline().split(',')]

        def filter_fn(x): return x.endswith('.csv') and 'Names' not in x
        csv_files = list(filter(filter_fn, os.listdir(work_folder)))

        for base_filename in csv_files:
            filename_csv = os.path.join(work_folder, base_filename)
            filename_parquet = filename_csv.replace('.csv', '.parquet')
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df = pd.read_csv(filename_csv, header=None,
                             names=columns, low_memory=False)
            df[self.target] = df['category'] + '_' + df['subcategory']
            df = df.drop(columns=['pkSeqID', 'stime', 'saddr', 'daddr', 'seq',
                                  'attack', 'category', 'subcategory'])
            df.to_parquet(filename_parquet)
            log_print(f'Converted file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'Entire Dataset')

        def filter_fn(x): return x.endswith('.parquet')
        parquet_files = list(filter(filter_fn, os.listdir(work_folder)))
        data_frames = []
        for base_filename in parquet_files:
            log_print(f'Loading parquet file \'{base_filename}\'.')
            full_filename = os.path.join(work_folder, base_filename)
            data_frames.append(pd.read_parquet(full_filename))
            log_print(f'Loaded parquet file \'{base_filename}\'.')
        self.data = pd.concat(data_frames, copy=False)
