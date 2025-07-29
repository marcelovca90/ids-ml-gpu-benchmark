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

class CICEVSE2024(BasePreprocessingPipeline):

    def __init__(self, subfolder=None, mode=None, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.base_folder = os.path.join('datasets', 'CICEVSE2024')
        self.base_name = 'CICEVSE2024'
        self.subfolder = subfolder
        self.mode = mode
        self.name = f'{self.base_name}_{subfolder}_{self.mode.title()}_{self.kind}'
        self.folder = self.base_folder
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        log_print(f'Processing subfolder \'{self.subfolder}\' and saving to parquet.')
        subfolder_path = os.path.join(self.base_folder, 'source', self.subfolder)
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        cols_to_drop = [
            "id", "expiration_id", "src_ip", "dst_ip", "src_mac", "dst_mac",
            "src_oui", "dst_oui", "src_port", "requested_server_name",
            "client_fingerprint", "server_fingerprint",
            "user_agent", "content_type", "tunnel_id", "vlan_id"
        ]

        curr_folder_dfs = []
        for csv_file in tqdm(csv_files, desc='CSV', leave=False):
            csv_path = os.path.join(subfolder_path, csv_file)
            curr_df = pd.read_csv(csv_path, low_memory=False)
            curr_df = curr_df.drop(columns=cols_to_drop, errors='ignore')
            base_label = Path(csv_file).stem.replace(f'{self.subfolder}-', '').lower()
            if self.mode == 'micro':
                curr_label = base_label
            elif self.mode == 'macro':
                curr_label = base_label.replace('charging-', '').replace('idle-', '')
            curr_df[self.target] = curr_label
            curr_folder_dfs.append(curr_df)

        df = pd.concat(curr_folder_dfs)
        
        if self.binarize:
            if self.mode == 'micro':
                df[self.target] = np.where(df[self.target] in ['idle-benign', 'charging-benign'], 'Benign', 'Malign')
            elif self.mode == 'macro':
                df[self.target] = np.where(df[self.target] == 'benign', 'Benign', 'Malign')

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

# PYTHONPATH=. python modules/preprocessing/custom/CICEVSE2024.py
if __name__ == "__main__":

    binarize_flags = [False]

    subfolders = ['EVSE-A', 'EVSE-B']
    
    modes = ['micro', 'macro']

    for i, binarize_flag in enumerate(tqdm(binarize_flags, desc="Binarize", leave=False)):

        for j, subfolder in enumerate(tqdm(subfolders, desc="Subfolder", leave=False)):

            for k, mode in enumerate(tqdm(modes, desc="Mode", leave=False)):

                try:

                    msg_prefix = f"[{i+1:02}/{len(binarize_flags):02}] [{j+1:02}/{len(subfolders):02} [{k+1:02}/{len(modes):02}]"

                    log_print(f'{msg_prefix} Started processing CICEVSE2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')
                    post_disc(f'{msg_prefix} Started processing CICEVSE2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')

                    subfolder_path = os.path.join('datasets/CICEVSE2024/source', subfolder)

                    cicevse = CICEVSE2024(subfolder=subfolder, mode=mode, binarize=binarize_flag)

                    cicevse.pipeline()

                    log_print(f'{msg_prefix} Finished processing CICEVSE2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')
                    post_disc(f'{msg_prefix} Finished processing CICEVSE2024/{subfolder} (binarize={binarize_flag}, mode={mode}).')

                except Exception as e:

                    log_print(f'{msg_prefix} Error processing CICEVSE2024/{subfolder} (binarize={binarize_flag}, mode={mode}): {str(e)}')
                    post_disc(f'{msg_prefix} Error processing CICEVSE2024/{subfolder} (binarize={binarize_flag}, mode={mode}): {str(e)}')
