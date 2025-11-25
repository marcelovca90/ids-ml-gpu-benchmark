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

class N_BaIoT(BasePreprocessingPipeline):

    def __init__(self, sample_frac, seed, binarize, subfolder=None,) -> None:
        super().__init__(sample_frac=sample_frac, seed=seed, binarize=binarize)
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

# PYTHONPATH=. python modules/preprocessing/custom/n_baiot.py
if __name__ == "__main__":

    # For logging
    errors = []

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

    # Calculate total steps for progress bar prefix
    total_steps = len(BINARIZE_FLAGS) * len(subfolders)

    # 1. Loop Configuration (Binarize)
    for i, binarize_flag in enumerate(tqdm(BINARIZE_FLAGS, desc="Binarize", leave=False)):

        # 2. Loop Subfolders (Devices)
        for j, subfolder in enumerate(tqdm(subfolders, desc="N_BaIoT_Device", leave=False)):

            # Construct readable ID and progress prefix
            step_idx = (i * len(subfolders)) + j + 1
            msg_prefix = f"[{step_idx:02}/{total_steps:02}]"

            dataset_identifier = f"N_BaIoT/{subfolder}"

            # 3. Loop Seeds (Randomness)
            for seed in tqdm(SEEDS, desc='Seed', leave=False):

                # ==================================================
                # A. FULL RUN (Generator)
                # ==================================================
                suffix_full = f"binarize={binarize_flag} sample_frac=1.0 seed={seed}"

                run_result = safe_exec(
                    runnable=lambda: N_BaIoT(
                        subfolder=subfolder,
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
                        runnable=lambda: N_BaIoT(
                            subfolder=subfolder,
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

    # 4. Final Cleanup
    run_result = safe_exec(
        runnable=lambda: copy_files(),
        msg_prefix="[FINAL]",
        dataset_name="N_BaIoT",
        msg_suffix="Copying Files"
    )
    if not run_result['success']:
        errors.append((now(), "copy_files", run_result))

    # --- 5. Error Logging ----
    if errors:
        pprint(errors, indent=4)
        with open(f'logs/{now()}_N_BaIoT_errors.json', 'w') as f:
            json.dump(errors, f, indent=4, default=str)
