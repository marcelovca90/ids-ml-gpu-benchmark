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

class KDD_Cup_1999(BasePreprocessingPipeline):

    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'kdd_cup_1999')
        self.name = 'KDD_Cup_1999'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')

        filename_cols = os.path.join(work_folder, 'kddcup.names')
        columns = []
        with open(filename_cols) as file:
            columns = [re.sub(r':.*\n', '', x) for x in file.readlines()[1:]]
        columns.append('label')

        filename_csv = os.path.join(work_folder, 'kddcup.data.corrected')
        filename_parquet = os.path.join(work_folder, f'{self.name}.parquet')
        log_print(f'Converting file \'{filename_csv}\' to parquet.')
        df = pd.read_csv(filename_csv, header=None,
                         names=columns, low_memory=False)
        df.to_parquet(filename_parquet)
        log_print(f'Converted  file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        base_filename = os.path.join(work_folder, f'{self.name}.parquet')
        log_print(f'Loading parquet file \'{base_filename}\'.')
        full_filename = os.path.join(work_folder, base_filename)
        self.data = pd.read_parquet(full_filename)
        log_print(f'Loaded parquet file \'{base_filename}\'.')

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        for value in self.data[self.target].unique():
            _replace_values(self.data, self.target,
                            value, value.replace('.', ''))
        if self.binarize:
            self.data[self.target] = np.where(
                (self.data[self.target] == 'normal'), 'Benign', 'Malign'
            )
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)
