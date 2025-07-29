import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline

sys.path.append(Path(__file__).absolute().parent.parent)

class CICIoMT2024_WiFi_and_MQTT(BasePreprocessingPipeline):

    @function_call_logger
    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'CICIoMT2024')
        self.name = 'CICIoMT2024_WiFi_and_MQTT'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source/WiFI_and_MQTT/CSV')
        csv_files = glob.glob(os.path.join(work_folder, '**', '*.csv'), recursive=True)

        df_list = []
        for csv_filename in csv_files:
            log_print(f'Started processing CSV \'{csv_filename}\'.')
            label = re.match(r"^(.*?)(?:\d+)?(?=_test|_train)", Path(csv_filename).stem).group(1)
            df = pd.read_csv(csv_filename, low_memory=False)
            df['label'] = label
            log_print(f'Finished processing CSV \'{csv_filename}\'.')
            df_list.append(df)

        ans = pd.concat(df_list).infer_objects()

        if self.binarize:
            ans['label'] = np.where(
                (ans['label'] == 'Benign'), 'Benign', 'Malign'
            )

        filename_parquet = os.path.join(os.getcwd(), self.folder, f'source/WiFI_and_MQTT/PARQUET/{self.name}.parquet')
        os.makedirs(Path(filename_parquet).parent, exist_ok=True)
        log_print(f'Saving file \'{filename_parquet}\' to parquet.')
        ans.to_parquet(filename_parquet)
        log_print(f'Saved  file \'{filename_parquet}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source/WiFI_and_MQTT/PARQUET')
        full_filename = os.path.join(work_folder, f'{self.name}.parquet')
        log_print(f'Loading parquet file \'{full_filename}\'.')
        self.data = pd.read_parquet(full_filename)
        log_print(f'Loaded parquet file \'{full_filename}\'.')
