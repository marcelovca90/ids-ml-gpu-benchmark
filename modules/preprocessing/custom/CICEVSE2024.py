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

class CICEVSE2024(BasePreprocessingPipeline):

    def __init__(self, sample_frac, seed, binarize, subfolder=None, mode=None) -> None:
        super().__init__(sample_frac=sample_frac, seed=seed, binarize=binarize)
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

    subfolders = ['EVSE-A', 'EVSE-B']
    modes = ['micro', 'macro']

    # Calculate total steps for progress bar prefix
    total_steps = len(BINARIZE_FLAGS) * len(subfolders) * len(modes)

    # 1. Loop Configuration (Binarize)
    for i, binarize_flag in enumerate(tqdm(BINARIZE_FLAGS, desc="Binarize", leave=False)):

        # 2. Loop Subfolders
        for j, subfolder in enumerate(tqdm(subfolders, desc="Subfolder", leave=False)):

            # 3. Loop Modes
            for k, mode in enumerate(tqdm(modes, desc="Mode", leave=False)):

                # Construct readable ID and progress prefix
                step_idx = (i * len(subfolders) * len(modes)) + (j * len(modes)) + k + 1
                msg_prefix = f"[{step_idx:02}/{total_steps:02}]"
                
                dataset_identifier = f"CICEVSE2024/{subfolder}/{mode}"

                # 4. Loop Seeds (Randomness)
                for seed in tqdm(SEEDS, desc='Seed', leave=False):

                    # ==================================================
                    # A. FULL RUN (Generator)
                    # ==================================================
                    suffix_full = f"binarize={binarize_flag} sample_frac=1.0 seed={seed}"

                    success = safe_exec(
                        runnable=lambda: CICEVSE2024(
                            subfolder=subfolder,
                            mode=mode,
                            sample_frac=1.0,
                            seed=seed,
                            binarize=binarize_flag
                        ).pipeline(preload=False),
                        msg_prefix=msg_prefix,
                        dataset_name=dataset_identifier,
                        msg_suffix=suffix_full
                    )

                    # If Full run fails, skip sampled runs for this seed
                    if not success:
                        continue

                    # ==================================================
                    # B. SAMPLED RUNS (Consumers)
                    # ==================================================
                    for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):
                        if sample_frac == 1.0: 
                            continue 

                        suffix_sampled = f"binarize={binarize_flag} sample_frac={sample_frac} seed={seed}"

                        safe_exec(
                            runnable=lambda: CICEVSE2024(
                                subfolder=subfolder,
                                mode=mode,
                                sample_frac=sample_frac,
                                seed=seed,
                                binarize=binarize_flag
                            ).pipeline(preload=True),
                            msg_prefix=msg_prefix,
                            dataset_name=dataset_identifier,
                            msg_suffix=suffix_sampled
                        )

    # 5. Final Cleanup
    safe_exec(
        runnable=lambda: copy_files(),
        msg_prefix="[FINAL]",
        dataset_name="CICEVSE2024",
        msg_suffix="Copying Files"
    )