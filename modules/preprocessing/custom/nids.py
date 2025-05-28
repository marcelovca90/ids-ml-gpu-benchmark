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

class NIDS(BasePreprocessingPipeline):

    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.base_folder = os.path.join('datasets', 'NIDS')
        self.base_name = 'NIDS'
        self.data = dict()
        self.folder = dict()
        self.name = list()
        self.target = dict()
        self.multiple = True

    @function_call_logger
    def prepare(self) -> None:
        root_dir = Path(self.base_folder)
        csv_filenames = list(root_dir.rglob("*.csv"))
        csv_filenames_data = [x for x in csv_filenames \
            if 'generated' not in str(x) and 'Feature' not in str(x)]
        for csv_filename in csv_filenames_data:
            name = f'{self.base_name}_{csv_filename.stem}_{self.kind}'
            self.name.append(name)
            self.folder[name] = self.base_folder
            self.target[name] = 'label'
            log_print(f'Converting file \'{csv_filename}\' to parquet.')
            df = pd.read_csv(csv_filename)
            assert 'Label' in df.columns and 'Attack' in df.columns
            if self.binarize:
                df[self.target[name]] = np.where(
                    (df['Attack'] == 'Benign'), 'Benign', 'Malign'
                )
            else:
                df[self.target[name]] = df['Attack']
            df = df.drop(columns=[
                'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 
                'Attack', 'Label', 'Dataset'], errors='ignore'
            )
            parquet_filename = str(csv_filename) \
                .replace('.csv', f'_{self.kind}.parquet')
            df.to_parquet(parquet_filename)
            log_print(f'Converted file \'{csv_filename}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        root_dir = Path(self.base_folder)
        parquet_files = list(root_dir.rglob(f"*_{self.kind}.parquet"))
        for parquet_filename in parquet_files:
            name = f'{self.base_name}_{parquet_filename.stem}'
            log_print(f'Loading parquet file \'{parquet_filename}\'.')
            self.data[name] = pd.read_parquet(parquet_filename)
            log_print(f'Loaded parquet file \'{parquet_filename}\'.')

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        for name, data in self.data.items():
            log_value_counts(data, self.target[name])
            data_obj = data.select_dtypes(['object'])
            data[data_obj.columns] = data_obj.apply(lambda x: x.str.strip())
            log_print('Value counts after sanitization:')
            log_value_counts(data, self.target[name])

    @function_call_logger
    def save(self) -> None:
        super().save(csv=False, parquet=False, metadata=True)