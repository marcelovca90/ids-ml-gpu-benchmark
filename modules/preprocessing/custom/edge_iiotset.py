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

# PYTHONPATH=. python modules/preprocessing/custom/edge_iiotset.py
class EDGE_IIOTSET(BasePreprocessingPipeline):

    def __init__(self, sample_frac, seed, binarize, csv_filename=None) -> None:
        super().__init__(sample_frac=sample_frac, seed=seed, binarize=binarize)
        self.base_folder = os.path.join('datasets', 'EDGE-IIOTSET')
        self.base_name = 'EDGE-IIOTSET'
        self.csv_filename = csv_filename
        self.name = f"{self.base_name}_{csv_filename.replace('-dataset.csv', '')}_{self.kind}"
        self.folder = self.base_folder
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        log_print(f'Processing CSV \'{self.csv_filename}\' and saving to parquet.')
        csv_filename_full = os.path.join(self.base_folder, 'source', self.csv_filename)

        df = pd.read_csv(csv_filename_full, low_memory=False)

        df = df.drop(columns=["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                            "arp.dst.proto_ipv4", "tcp.ack_raw", "tcp.payload", "tcp.options", 
                            "http.file_data", "http.request.full_uri", "http.request.uri.query", 
                            "http.referer", "http.request.full_uri",  "Attack_label"], errors="ignore")

        df = df.rename(columns={"Attack_type": self.target})

        if self.binarize:
            df[self.target] = np.where(df[self.target] == 'Normal', 'Benign', 'Malign')

        self.data = df

        parquet_filename = csv_filename_full.replace('.csv', f'_{self.kind}.parquet')
        self.data.to_parquet(parquet_filename)
        log_print(f'Processed  CSV \'{csv_filename}\' and saving to parquet.')

    @function_call_logger
    def load(self) -> None:
        csv_filename_full = os.path.join(self.base_folder, 'source', self.csv_filename)
        parquet_filename = csv_filename_full.replace('.csv', f'_{self.kind}.parquet')
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

# PYTHONPATH=. python modules/preprocessing/custom/edge_iiotset.py
if __name__ == "__main__":

    # For logging
    errors = []

    csv_filenames = [
        "ML-EdgeIIoT-dataset.csv",
        "DNN-EdgeIIoT-dataset.csv"
    ]

    # Calculate total steps for progress bar prefix
    total_steps = len(BINARIZE_FLAGS) * len(csv_filenames)

    # 1. Loop Configuration (Binarize)
    for i, binarize_flag in enumerate(tqdm(BINARIZE_FLAGS, desc="Binarize", leave=False)):

        # 2. Loop Datasets (CSV Files)
        for j, csv_filename in enumerate(tqdm(csv_filenames, desc="EDGE_IIOTSET_CSV", leave=False)):

            # Construct readable ID and progress prefix
            step_idx = (i * len(csv_filenames)) + j + 1
            msg_prefix = f"[{step_idx:02}/{total_steps:02}]"
            
            dataset_identifier = f"EDGE_IIOTSET/{csv_filename}"

            # 3. Loop Seeds (Randomness)
            for seed in tqdm(SEEDS, desc='Seed', leave=False):

                # ==================================================
                # A. FULL RUN (Generator)
                # ==================================================
                suffix_full = f"binarize={binarize_flag} sample_frac=1.0 seed={seed}"

                run_result = safe_exec(
                    runnable=lambda: EDGE_IIOTSET(
                        csv_filename=csv_filename,
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
                    errors.append((now(), csv_filename, binarize_flag, seed, "full", run_result))
                    continue

                # ==================================================
                # B. SAMPLED RUNS (Consumers)
                # ==================================================
                for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):
                    if sample_frac == 1.0: 
                        continue 

                    suffix_sampled = f"binarize={binarize_flag} sample_frac={sample_frac} seed={seed}"

                    run_result = safe_exec(
                        runnable=lambda: EDGE_IIOTSET(
                            csv_filename=csv_filename,
                            sample_frac=sample_frac,
                            seed=seed,
                            binarize=binarize_flag
                        ).pipeline(preload=True),
                        msg_prefix=msg_prefix,
                        dataset_name=dataset_identifier,
                        msg_suffix=suffix_sampled
                    )
                    if not run_result['success']:
                        errors.append((now(), csv_filename, binarize_flag, seed, sample_frac, run_result))

    # 4. Final Cleanup
    run_result = safe_exec(
        runnable=lambda: copy_files(),
        msg_prefix="[FINAL]",
        dataset_name="EDGE_IIOTSET",
        msg_suffix="Copying Files"
    )
    if not run_result['success']:
        errors.append((now(), "copy_files", run_result))

    # --- 5. Error Logging ----
    if errors:
        pprint(errors, indent=4)
        with open(f'logs/{now()}_EDGE_IIOTSET_errors.json', 'w') as f:
            json.dump(errors, f, indent=4, default=str)
