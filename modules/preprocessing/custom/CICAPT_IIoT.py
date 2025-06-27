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

class CICAPT_IIoT(BasePreprocessingPipeline):

    def __init__(self, subfolder=None, subfile=None, mode=None, binarize=False) -> None:
        super().__init__(binarize=binarize)
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

    binarize_flags = [False]

    subfolder_and_subfiles = [
        {'folder': 'Phase1', 'file': 'phase1_NetworkData.csv'},
        {'folder': 'Phase2', 'file': 'phase2_NetworkData.csv'},
    ]
         
    modes = ['micro', 'macro']

    for i, binarize_flag in enumerate(tqdm(binarize_flags, desc="Binarize", leave=False)):

        for j, subfolder_and_subfile in enumerate(tqdm(subfolder_and_subfiles, desc="Subfolder", leave=False)):

            subfolder = subfolder_and_subfile['folder']
            subfile = subfolder_and_subfile['file']

            for k, mode in enumerate(tqdm(modes, desc="Mode", leave=False)):

                try:

                    msg_prefix = f"[{i+1:02}/{len(binarize_flags):02}] [{j+1:02}/{len(subfolder_and_subfiles):02} [{k+1:02}/{len(modes):02}]"

                    log_print(f'{msg_prefix} Started processing CICAPT_IIoT/{subfolder} (binarize={binarize_flag}, mode={mode}).')
                    post_disc(f'{msg_prefix} Started processing CICAPT_IIoT/{subfolder} (binarize={binarize_flag}, mode={mode}).')

                    subfolder_path = os.path.join('datasets/CICAPT_IIoT/source', subfolder)

                    cicapt = CICAPT_IIoT(subfolder=subfolder, subfile=subfile, mode=mode, binarize=binarize_flag)

                    cicapt.pipeline()

                    log_print(f'{msg_prefix} Finished processing CICAPT_IIoT/{subfolder} (binarize={binarize_flag}, mode={mode}).')
                    post_disc(f'{msg_prefix} Finished processing CICAPT_IIoT/{subfolder} (binarize={binarize_flag}, mode={mode}).')

                except Exception as e:

                    log_print(f'{msg_prefix} Error processing CICAPT_IIoT/{subfolder} (binarize={binarize_flag}, mode={mode}): {str(e)}')
                    post_disc(f'{msg_prefix} Error processing CICAPT_IIoT/{subfolder} (binarize={binarize_flag}, mode={mode}): {str(e)}')
