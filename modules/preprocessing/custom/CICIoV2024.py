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

class CICIoV2024(BasePreprocessingPipeline):

    def __init__(self, subfolder=None, mode=None, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.base_folder = os.path.join('datasets', 'CICIoV2024')
        self.base_name = 'CICIoV2024'
        self.subfolder = subfolder
        self.mode = mode
        self.name = f'{self.base_name}_{subfolder.title()}_{self.mode.title()}_{self.kind}'
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
            curr_df = curr_df.drop(columns=['label'], errors='ignore')
            if self.mode == 'micro':
                curr_df[self.target] = curr_df['category'] + '_' + curr_df['specific_class']
            elif self.mode == 'macro':
                curr_df[self.target] = curr_df['category']
            curr_df = curr_df.drop(columns=['category', 'specific_class'], errors='ignore')
            curr_folder_dfs.append(curr_df)

        df = pd.concat(curr_folder_dfs)
        
        if self.binarize:
            if self.mode == 'micro':
                df[self.target] = np.where(df[self.target] == 'BENIGN_BENIGN', 'Benign', 'Malign')
            elif self.mode == 'macro':
                df[self.target] = np.where(df[self.target] == 'BENIGN', 'Benign', 'Malign')

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

# PYTHONPATH=. python modules/preprocessing/custom/CICIoV2024.py
if __name__ == "__main__":

    binarize_flags = [False]

    subfolders = ['binary', 'decimal'] #, 'hexadecimal']
    
    modes = ['micro', 'macro']

    for i, binarize_flag in enumerate(tqdm(binarize_flags, desc="Binarize", leave=False)):

        for j, subfolder in enumerate(tqdm(subfolders, desc="Subfolder", leave=False)):

            for k, mode in enumerate(tqdm(modes, desc="Mode", leave=False)):

                try:

                    msg_prefix = f"[{i+1:02}/{len(binarize_flags):02}] [{j+1:02}/{len(subfolders):02} [{k+1:02}/{len(modes):02}]"

                    log_print(f'{msg_prefix} Started processing CICIoV2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')
                    post_disc(f'{msg_prefix} Started processing CICIoV2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')

                    subfolder_path = os.path.join('datasets/CICIoV2024/source', subfolder)

                    ciciov = CICIoV2024(subfolder=subfolder, mode=mode, binarize=binarize_flag)

                    ciciov.pipeline()

                    log_print(f'{msg_prefix} Finished processing CICIoV2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')
                    post_disc(f'{msg_prefix} Finished processing CICIoV2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')

                except Exception as e:

                    log_print(f'{msg_prefix} Error processing CICIoV2024/{subfolder} (binarize={binarize_flag}, mode={mode}): {str(e)}')
                    post_disc(f'{msg_prefix} Error processing CICIoV2024/{subfolder} (binarize={binarize_flag}, mode={mode}): {str(e)}')
