import os
import sys
import warnings
from pathlib import Path

sys.path.append(Path(__file__).absolute().parent.parent)

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in reduce", category=RuntimeWarning)

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.logging.webhook import post_disc
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts
from tqdm import tqdm

class BCCC(BasePreprocessingPipeline):

    def __init__(self, subfolder=None, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.base_folder = os.path.join('datasets', 'BCCC')
        self.base_name = 'BCCC'
        self.subfolder = subfolder
        self.name = f'{self.base_name}_{subfolder.replace(" ", "_")}_{self.kind}'
        self.folder = self.base_folder
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        log_print(f'Processing subfolder \'{self.subfolder}\' and saving to parquet.')
        subfolder_path = os.path.join(self.base_folder, 'source', self.subfolder)
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        curr_folder_dfs = []
        for csv_file in tqdm(csv_files, desc='CSV', leave=False):
            csv_path = os.path.join(subfolder_path, csv_file)
            curr_df = pd.read_csv(csv_path, low_memory=False)
            curr_df = curr_df.drop(
                columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label'],
                errors='ignore'
            )
            curr_folder_dfs.append(curr_df)

        df = pd.concat(curr_folder_dfs).rename(columns={'Attack Name': self.target})

        if self.binarize:
            df[self.target] = np.where(df[self.target] == 'Benign Traffic', 'Benign', 'Malign')

        self.data = df

        parquet_filename = os.path.join(subfolder_path, f'Merged_{self.kind}.parquet')
        self.data.to_parquet(parquet_filename)
        log_print(f'Processed  subfolder \'{self.subfolder}\' and saving to parquet.')

    @function_call_logger
    def load(self) -> None:
        subfolder_path = os.path.join(self.base_folder, 'source', self.subfolder)
        parquet_filename = os.path.join(subfolder_path, f'Merged_{self.kind}.parquet')
        log_print(f'Loading parquet file \'{parquet_filename}\'.')
        self.data = pd.read_parquet(parquet_filename)
        log_print(f'Loaded parquet file \'{parquet_filename}\'.')

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        data_obj = self.data.select_dtypes(['object'])
        self.data[data_obj.columns] = data_obj.apply(lambda x: x.str.strip())
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)

# PYTHONPATH=. python modules/preprocessing/custom/bccc.py
if __name__ == "__main__":

    binarize_flags = [False]

    subfolders = [
        'CIC-BCCC-NRC-ACI-IOT-2023',
        'CIC-BCCC-NRC-Edge-IIoTSet-2022',
        'CIC-BCCC-NRC-IoMT-2024',
        'CIC-BCCC-NRC-IoT-2022',
        'CIC-BCCC-NRC-IoT-2023-Original Training and Testing',
        'CIC-BCCC-NRC-IoT-HCRL-2019',
        'CIC-BCCC-NRC-MQTTIoT-IDS-2020',
        'CIC-BCCC-NRC-TONIoT-2021',
        'CIC-BCCC-NRC-UQ-IOT-2022'
    ]

    for i, binarize_flag in enumerate(tqdm(binarize_flags, desc="Binarize", leave=False)):

        for j, subfolder in enumerate(tqdm(subfolders, desc="BCCC_CSV", leave=False)):

            try:

                msg_prefix = f"[{i+1:02}/{len(binarize_flags):02}] [{j+1:02}/{len(subfolders):02}]"

                log_print(f'{msg_prefix} Started processing BCCC/{subfolder} (binarize={binarize_flag}).')
                post_disc(f'{msg_prefix} Started processing BCCC/{subfolder} (binarize={binarize_flag}).')

                subfolder_path = os.path.join('datasets/BCCC/source', subfolder)

                bccc = BCCC(subfolder=subfolder, binarize=binarize_flag)

                bccc.pipeline()

                log_print(f'{msg_prefix} Finished processing BCCC/{subfolder} (binarize={binarize_flag}).')
                post_disc(f'{msg_prefix} Finished processing BCCC/{subfolder} (binarize={binarize_flag}).')

            except Exception as e:

                log_print(f'{msg_prefix} Error processing BCCC/{subfolder} (binarize={binarize_flag}): {str(e)}')
                post_disc(f'{msg_prefix} Error processing BCCC/{subfolder} (binarize={binarize_flag}): {str(e)}')
