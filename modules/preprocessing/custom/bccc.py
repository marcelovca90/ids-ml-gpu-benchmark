import os
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
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts

class BCCC(BasePreprocessingPipeline):

    def __init__(self, sample_frac, seed, binarize, subfolder=None) -> None:
        super().__init__(sample_frac=sample_frac, seed=seed, binarize=binarize)
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

    # For progress bar calculation
    total_steps = len(BINARIZE_FLAGS) * len(subfolders)
    
    # 1. Loop Configuration (Binarize)
    for i, binarize_flag in enumerate(tqdm(BINARIZE_FLAGS, desc="Binarize", leave=False)):
        
        # 2. Loop Datasets (Subfolders)
        for j, subfolder in enumerate(tqdm(subfolders, desc="BCCC_Subfolder", leave=False)):
            
            # Construct a readable ID for logs
            step_idx = (i * len(subfolders)) + j + 1
            msg_prefix = f"[{step_idx:02}/{total_steps:02}]"
            dataset_identifier = f"BCCC/{subfolder}"

            # 3. Loop Seeds (Randomness)
            for seed in tqdm(SEEDS, desc='Seed', leave=False):
                
                # ==================================================
                # A. FULL RUN (Generator)
                # ==================================================
                # Must run first with sample_frac=1.0 and preload=False
                # to generate the base artifacts (cleaning, splitting, ID creation).
                suffix_full = f"binarize={binarize_flag} sample_frac=1.0 seed={seed}"
                
                success = safe_exec(
                    runnable=lambda: BCCC(
                        subfolder=subfolder, 
                        sample_frac=1.0, 
                        seed=seed, 
                        binarize=binarize_flag
                    ).pipeline(preload=False),
                    msg_prefix=msg_prefix,
                    dataset_name=dataset_identifier,
                    msg_suffix=suffix_full
                )

                # Critical Safety Check:
                # If the Full run fails (missing raw CSV, etc.), 
                # we MUST skip sampled runs for this seed as they have nothing to load.
                if not success:
                    continue

                # ==================================================
                # B. SAMPLED RUNS (Consumers)
                # ==================================================
                # Iterate through fractions, skip 1.0 (done above), and use preload=True
                for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):
                    if sample_frac == 1.0: 
                        continue 

                    suffix_sampled = f"binarize={binarize_flag} sample_frac={sample_frac} seed={seed}"

                    safe_exec(
                        runnable=lambda: BCCC(
                            subfolder=subfolder, 
                            sample_frac=sample_frac, 
                            seed=seed, 
                            binarize=binarize_flag
                        ).pipeline(preload=True),
                        msg_prefix=msg_prefix,
                        dataset_name=dataset_identifier,
                        msg_suffix=suffix_sampled
                    )

    # 4. Final Cleanup / Organization
    safe_exec(
        runnable=lambda: copy_files(),
        msg_prefix="[FINAL]",
        dataset_name="BCCC",
        msg_suffix="Copying Files"
    )