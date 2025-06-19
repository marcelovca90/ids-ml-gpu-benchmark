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

class N_BaIoT(BasePreprocessingPipeline):

    def __init__(self, subfolder=None, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.base_folder = os.path.join('datasets', 'N_BaIoT')
        self.base_name = 'N_BaIoT'
        self.subfolder = subfolder
        self.name = f'{self.base_name}_{subfolder}_{self.kind}'
        self.folder = self.base_folder
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        log_print(f'Processing subfolder \'{self.subfolder}\' and saving to parquet.')
        subfolder_path = os.path.join(self.base_folder, 'source', self.subfolder)

        dfs = []
        
        benign_df = pd.read_csv(os.path.join(subfolder_path, 'benign_traffic.csv'))
        benign_df[self.target] = 'benign_traffic'
        dfs.append(benign_df)

        malign_botnets = ['gafgyt_attacks', 'mirai_attacks']
        for malign_botnet in malign_botnets:
            malign_subfolder = os.path.join(subfolder_path, malign_botnet)
            if os.path.exists(malign_subfolder):
                csv_files = [f for f in os.listdir(malign_subfolder) if f.endswith('.csv')]
                for csv_file in tqdm(csv_files, desc='CSV', leave=False):
                    malign_df_tmp = pd.read_csv(os.path.join(malign_subfolder, csv_file))
                    malign_df_tmp[self.target] = f"{malign_botnet.replace('_attacks', '')}_{csv_file.replace('.csv', '')}"
                    dfs.append(malign_df_tmp)
        
        df = pd.concat(dfs, axis='index')

        if self.binarize:
            df[self.target] = np.where(df[self.target] == 'benign_traffic', 'Benign', 'Malign')

        self.data = df

        parquet_filename = os.path.join(subfolder_path, f'Merged_{self.kind}.parquet')
        self.data.to_parquet(parquet_filename)
        log_print(f'Processed  subfolder \'{subfolder}\' and saving to parquet.')

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

# PYTHONPATH=. python modules/preprocessing/custom/n_baiot.py
if __name__ == "__main__":

    binarize_flags = [False, True]

    subfolders = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera"
    ]

    for i, binarize_flag in enumerate(tqdm(binarize_flags, desc="Binarize", leave=False)):

        for j, subfolder in enumerate(tqdm(subfolders, desc="N_BaIoT_CSV", leave=False)):

            try:

                msg_prefix = f"[{i+1:02}/{len(binarize_flags):02}] [{j+1:02}/{len(subfolders):02}]"

                log_print(f'{msg_prefix} Started processing N_BaIoT/{subfolder} (binarize={binarize_flag}).')
                post_disc(f'{msg_prefix} Started processing N_BaIoT/{subfolder} (binarize={binarize_flag}).')

                subfolder_path = os.path.join('datasets/N_BaIoT/source', subfolder)

                nbaiot = N_BaIoT(subfolder=subfolder, binarize=binarize_flag)

                nbaiot.pipeline(preload=False, shrink_mode=None, complexity_mode=None, profile_mode='minimal')

                log_print(f'{msg_prefix} Finished processing N_BaIoT/{subfolder} (binarize={binarize_flag}).')
                post_disc(f'{msg_prefix} Finished processing N_BaIoT/{subfolder} (binarize={binarize_flag}).')

            except Exception as e:

                log_print(f'{msg_prefix} Error processing N_BaIoT/{subfolder} (binarize={binarize_flag}): {str(e)}')
                post_disc(f'{msg_prefix} Error processing N_BaIoT/{subfolder} (binarize={binarize_flag}): {str(e)}')
