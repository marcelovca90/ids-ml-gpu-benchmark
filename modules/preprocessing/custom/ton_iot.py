import os
import re
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

sys.path.append(Path(__file__).absolute().parent.parent)

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in reduce", category=RuntimeWarning)

from constants import BINARIZE_FLAGS, SAMPLE_FRACS, SEEDS
from modules.preprocessing.preproc_utils import safe_exec
from modules.filesystem.file_utils import copy_files

from modules.logging.logger import function_call_logger, log_print
from modules.logging.webhook import post_disc
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts
from modules.preprocessing.preproc_utils import _clean_and_expand_kmg_suffix

class ToN_IoT(BasePreprocessingPipeline):

    def __init__(self, sample_frac, seed, binarize, sub_name=None, sub_config=None) -> None:
        super().__init__(sample_frac=sample_frac, seed=seed, binarize=binarize)
        self.base_folder = os.path.join('datasets', 'ToN_IoT')
        self.base_name = 'ToN_IoT'
        self.sub_name = sub_name
        self.sub_config = sub_config
        self.name = f'{self.base_name}_{self.sub_name}_{self.kind}'
        self.folder = self.base_folder
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        log_print(f'Processing dataset \'{self.sub_name}\' and saving to parquet.')
        subfolder_path = os.path.join(self.base_folder, 'source', self.sub_config['folder'])
        # cols_to_drop = ['date', 'time', 'ts', 'PID', 'TRUN', 'src_ip', 'dst_ip',
        #                 'checksum', 'weird_name', 'weird_addl', 'weird_notice', 'label']
        cols_to_drop = ['date', 'time', 'ts', 'src_ip', 'dst_ip', 'label']
        
        if self.sub_config['mode'] == 'single':
            assert len(self.sub_config['files']) == 1
            csv_path = os.path.join(subfolder_path, self.sub_config['files'][0])
            df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
            df = df.drop(columns=cols_to_drop, errors='ignore')
            df = df.rename(columns={'type': self.target})
            if self.binarize:
                df[self.target] = np.where(df[self.target] == 'normal', 'Benign', 'Malign')
            self.data = df
            parquet_filename = os.path.join(subfolder_path, f'{self.sub_name}_{self.kind}.parquet')
            self.data.to_parquet(parquet_filename)
            log_print(f'Processed  dataset \'{self.sub_name}\' and saving to parquet.')
        
        elif self.sub_config['mode'] == 'multi':
            assert len(self.sub_config['files']) > 1
            csv_files = [os.path.join(subfolder_path, f) for f in self.sub_config['files']]
            curr_dfs = []
            for csv_file in tqdm(csv_files, desc='CSV', leave=False):
                curr_df = pd.read_csv(csv_file, low_memory=False, on_bad_lines="skip", sep=',')
                curr_df = curr_df.drop(columns=cols_to_drop, errors='ignore')
                for col in curr_df.columns:
                    curr_df[col] = curr_df[col].replace({'-': '0.0'})
                    if col in ['POLI', 'Status']:
                        curr_df[col] = pd.factorize(curr_df[col].astype(str))[0]
                    elif col in ['VGROW', 'RGROW']:
                        curr_df[col] = curr_df[col].apply(lambda x : re.sub(r'\s\d+', '', str(x)))
                    elif col in ['WRDSK', 'RDDSK', 'DSK']:
                        curr_df[col] = pd.to_numeric(curr_df[col], errors='coerce').dropna()
                    elif col in ['WCANCL', 'MINFLT', 'MAJFLT', 'VSTEXT', 'RSIZE']:
                        curr_df[col] = curr_df[col].astype(str).apply(_clean_and_expand_kmg_suffix).astype(float)
                curr_dfs.append(curr_df)
            df = pd.concat(curr_dfs).rename(columns={'type': self.target})
            if self.binarize:
                df[self.target] = np.where(df[self.target] == 'normal', 'Benign', 'Malign')
            self.data = df
            parquet_filename = os.path.join(subfolder_path, f'{self.sub_name}_{self.kind}.parquet')
            self.data.to_parquet(parquet_filename)
            log_print(f'Processed  dataset \'{self.sub_name}\' and saving to parquet.')

    @function_call_logger
    def load(self) -> None:
        subfolder_path = os.path.join(self.base_folder, 'source', self.sub_config['folder'])
        parquet_filename = os.path.join(subfolder_path, f'{self.sub_name}_{self.kind}.parquet')
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

# PYTHONPATH=. python modules/preprocessing/custom/ton_iot.py
if __name__ == "__main__":

    configs = {
        'IoT_Fridge': {
            'folder': 'Processed_IoT_dataset',
            'mode': 'single',
            'files': ['IoT_Fridge.csv']
        },
        'IoT_GPS_Tracker': {
            'folder': 'Processed_IoT_dataset',
            'mode': 'single',
            'files': ['IoT_GPS_Tracker.csv']
        },
        'IoT_Garage_Door': {
            'folder': 'Processed_IoT_dataset',
            'mode': 'single',
            'files': ['IoT_Garage_Door.csv']
        },
        'IoT_Modbus': {
            'folder': 'Processed_IoT_dataset',
            'mode': 'single',
            'files': ['IoT_Modbus.csv']
        },
        'IoT_Motion_Light': {
            'folder': 'Processed_IoT_dataset',
            'mode': 'single',
            'files': ['IoT_Motion_Light.csv']
        },
        'IoT_Thermostat': {
            'folder': 'Processed_IoT_dataset',
            'mode': 'single',
            'files': ['IoT_Thermostat.csv']
        },
        'IoT_Weather': {
            'folder': 'Processed_IoT_dataset',
            'mode': 'single',
            'files': ['IoT_Weather.csv']
        },
        'Linux_Disk': {
            'folder': 'Processed_Linux_dataset',
            'mode': 'multi',
            'files': [
                'linux_disk_1.csv',
                'linux_disk_2.csv'
            ]
        },
        'Linux_Memory': {
            'folder': 'Processed_Linux_dataset',
            'mode': 'multi',
            'files': [
                'linux_memory1.csv',
                'linux_memory2.csv'
            ]
        },
        'Linux_Process': {
            'folder': 'Processed_Linux_dataset',
            'mode': 'multi',
            'files': [
                'Linux_process_1.csv',
                'Linux_process_2.csv'
            ]
        },
        'Network': {
            'folder': 'Processed_Network_dataset',
            'mode': 'multi',
            'files': [
                f'Network_dataset_{i}.csv' for i in range(1, 24) # Refactored list comprehension
            ]
        },
        'Windows_7': {
            'folder': 'Processed_Windows_dataset',
            'mode': 'single',
            'files': [
                'windows7_dataset.csv'
            ]
        },
        'Windows_10': {
            'folder': 'Processed_Windows_dataset',
            'mode': 'single',
            'files': [
                'windows10_dataset.csv'
            ]
        }
    }

    # Calculate total steps
    total_steps = len(BINARIZE_FLAGS) * len(configs)

    # 1. Loop Configuration (Binarize)
    for i, binarize_flag in enumerate(tqdm(BINARIZE_FLAGS, desc="Binarize", leave=False)):

        # 2. Loop Datasets (Config Items)
        for j, (name, config) in enumerate(tqdm(configs.items(), desc="ToN_IoT_Config", leave=False)):

            # Construct readable ID and progress prefix
            step_idx = (i * len(configs)) + j + 1
            msg_prefix = f"[{step_idx:02}/{total_steps:02}]"
            
            dataset_identifier = f"ToN_IoT/{name}"

            # 3. Loop Seeds (Randomness)
            for seed in tqdm(SEEDS, desc='Seed', leave=False):

                # ==================================================
                # A. FULL RUN (Generator)
                # ==================================================
                suffix_full = f"binarize={binarize_flag} sample_frac=1.0 seed={seed}"

                run_result = safe_exec(
                    runnable=lambda: ToN_IoT(
                        sub_name=name,
                        sub_config=config,
                        sample_frac=1.0,
                        seed=seed,
                        binarize=binarize_flag
                    ).pipeline(preload=False),
                    msg_prefix=msg_prefix,
                    dataset_name=dataset_identifier,
                    msg_suffix=suffix_full
                )

                # If Full run fails, skip sampled runs for this seed
                if not run_result['success']:
                    continue

                # ==================================================
                # B. SAMPLED RUNS (Consumers)
                # ==================================================
                for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):
                    if sample_frac == 1.0: 
                        continue 

                    suffix_sampled = f"binarize={binarize_flag} sample_frac={sample_frac} seed={seed}"

                    run_result = safe_exec(
                        runnable=lambda: ToN_IoT(
                            sub_name=name,
                            sub_config=config,
                            sample_frac=sample_frac,
                            seed=seed,
                            binarize=binarize_flag
                        ).pipeline(preload=True),
                        msg_prefix=msg_prefix,
                        dataset_name=dataset_identifier,
                        msg_suffix=suffix_sampled
                    )

    # 4. Final Cleanup
    run_result = safe_exec(
        runnable=lambda: copy_files(),
        msg_prefix="[FINAL]",
        dataset_name="ToN_IoT",
        msg_suffix="Copying Files"
    )