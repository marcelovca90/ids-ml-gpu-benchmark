import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline

sys.path.append(Path(__file__).absolute().parent.parent)

class CIC_IDS_2017(BasePreprocessingPipeline):

    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'CIC_IDS_2017')
        self.name = 'CIC_IDS_2017'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'CSVs', 'MachineLearningCVE'
        )

        def filter_fn(x): return x.endswith('.csv') and 'Names' not in x
        csv_files = list(filter(filter_fn, os.listdir(work_folder)))

        for base_filename in csv_files:
            filename_csv = os.path.join(work_folder, base_filename)
            filename_parquet = filename_csv.replace('.csv', '.parquet')
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df = pd.read_csv(filename_csv, low_memory=False)
            df = df.rename(columns={v: str(v).strip() for v in df.columns})
            df = df.rename(columns={'Label': self.target})
            if self.binarize:
                df[self.target] = np.where(
                    (df[self.target] == 'BENIGN'), 'Benign', 'Malign'
                )
            df.to_parquet(filename_parquet)
            log_print(f'Converted  file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(
            os.getcwd(), self.folder, 'source', 'CSVs', 'MachineLearningCVE'
        )

        def filter_fn(x): return x.endswith('.parquet')
        parquet_files = list(filter(filter_fn, os.listdir(work_folder)))

        data_frames = []
        for base_filename in parquet_files:
            log_print(f'Loading parquet file \'{base_filename}\'.')
            full_filename = os.path.join(work_folder, base_filename)
            data_frames.append(pd.read_parquet(full_filename))
            log_print(f'Loaded parquet file \'{base_filename}\'.')
        self.data = pd.concat(data_frames, copy=False)
