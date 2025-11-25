import json
import os
import sys
import warnings
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd

sys.path.append(Path(__file__).absolute().parent.parent)

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in reduce", category=RuntimeWarning)

from modules.preprocessing.constants import BINARIZE_FLAGS, SAMPLE_FRACS, SEEDS
from modules.preprocessing.preproc_utils import now, safe_exec
from modules.filesystem.file_utils import copy_files

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts

class CICAPT_IIoT(BasePreprocessingPipeline):

    def __init__(self, sample_frac, seed, binarize, subfolder=None, subfile=None, mode=None) -> None:
        super().__init__(sample_frac=sample_frac, seed=seed, binarize=binarize)
        self.base_folder = os.path.join('datasets', 'CICAPT_IIoT')
        self.base_name = 'CICAPT_IIoT'
        self.subfolder = subfolder
        self.subfile = subfile
        self.mode = mode
        self.name = f'{self.base_name}_{subfolder.title()}_{self.mode.title()}_{self.kind}'
        self.folder = self.base_folder
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        log_print(f'Processing subfile \'{self.subfile}\' and saving to parquet.')
        subfolder_path = os.path.join(self.base_folder, 'source', self.subfolder)
        cols_to_drop = ["ts", "Sequence number", "Source IP",
                        "Destination IP", "Source Port", "MAC", "label"]

        csv_path = os.path.join(subfolder_path, self.subfile)
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.drop(columns=cols_to_drop, errors='ignore')
        df = df.drop(columns=['label'], errors='ignore')
        if self.mode == 'micro':
            df[self.target] = df['subLabel'].replace({0: 'benign', '0': 'benign'})
        elif self.mode == 'macro':
            df[self.target] = df['subLabelCat'].replace({0: 'benign', '0': 'benign'})
        df = df.drop(columns=['subLabel', 'subLabelCat'], errors='ignore')

        if self.binarize:
            df[self.target] = np.where(df[self.target] == 'benign', 'Benign', 'Malign')

        self.data = df

        parquet_filename = os.path.join(subfolder_path, f'Merged_{self.kind}.parquet')
        self.data.to_parquet(parquet_filename)
        log_print(f'Processed  subfile \'{self.subfile}\' and saving to parquet.')

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
        self.data[self.target] = self.data[self.target] \
            .astype(str).str.replace(r"[^A-Za-z0-9]", "_", regex=True) \
            .astype("category")
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)

# PYTHONPATH=. python modules/preprocessing/custom/CICAPT_IIoT.py
if __name__ == "__main__":

    # For logging
    errors = []

    subfolder_and_subfiles = [
        # {'folder': 'Phase1', 'file': 'phase1_NetworkData.csv'}, # single class
        {'folder': 'Phase2', 'file': 'phase2_NetworkData.csv'},
    ]

    modes = ['micro', 'macro']

    # Calculate total steps for the progress bar prefix
    total_steps = len(BINARIZE_FLAGS) * len(subfolder_and_subfiles) * len(modes)

    # 1. Loop Configuration (Binarize)
    for i, binarize_flag in enumerate(tqdm(BINARIZE_FLAGS, desc="Binarize", leave=False)):

        # 2. Loop Subfolders (Dataset Config Part A)
        for j, config in enumerate(tqdm(subfolder_and_subfiles, desc="Subfolder", leave=False)):
            subfolder = config['folder']
            subfile = config['file']

            # 3. Loop Modes (Dataset Config Part B)
            for k, mode in enumerate(tqdm(modes, desc="Mode", leave=False)):

                # Construct readable ID and progress prefix
                # Math: (Current Binarize Block) + (Current Subfolder Block) + (Current Mode)
                step_idx = (i * len(subfolder_and_subfiles) * len(modes)) + (j * len(modes)) + k + 1
                msg_prefix = f"[{step_idx:02}/{total_steps:02}]"

                dataset_identifier = f"CICAPT/{subfolder}/{mode}"

                # 4. Loop Seeds (Randomness)
                for seed in tqdm(SEEDS, desc='Seed', leave=False):

                    # ==================================================
                    # A. FULL RUN (Generator)
                    # ==================================================
                    suffix_full = f"binarize={binarize_flag} sample_frac=1.0 seed={seed}"

                    run_result = safe_exec(
                        runnable=lambda: CICAPT_IIoT(
                            subfolder=subfolder,
                            subfile=subfile,
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
                    if not run_result['success']:
                        errors.append((now(), subfolder, binarize_flag, seed, "full", run_result))
                        continue

                    # ==================================================
                    # B. SAMPLED RUNS (Consumers)
                    # ==================================================
                    for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):
                        if sample_frac == 1.0: 
                            continue 

                        suffix_sampled = f"binarize={binarize_flag} sample_frac={sample_frac} seed={seed}"

                        run_result = safe_exec(
                            runnable=lambda: CICAPT_IIoT(
                                subfolder=subfolder,
                                subfile=subfile,
                                mode=mode,
                                sample_frac=sample_frac,
                                seed=seed,
                                binarize=binarize_flag
                            ).pipeline(preload=True),
                            msg_prefix=msg_prefix,
                            dataset_name=dataset_identifier,
                            msg_suffix=suffix_sampled
                        )
                        if not run_result['success']:
                            errors.append((now(), subfolder, binarize_flag, seed, sample_frac, run_result))

     # 5. Final Cleanup / Organization
    run_result = safe_exec(
        runnable=lambda: copy_files(),
        msg_prefix="[FINAL]",
        dataset_name="CICAPT_IIoT",
        msg_suffix="Copying Files"
    )
    if not run_result['success']:
        errors.append((now(), "copy_files", run_result))

    # --- 6. Error Logging ----
    if errors:
        pprint(errors, indent=4)
        with open(f'logs/{now()}_CICAPT_IIoT_errors.json', 'w') as f:
            json.dump(errors, f, indent=4, default=str)
