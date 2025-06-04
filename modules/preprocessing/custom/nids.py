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
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_value_counts
from tqdm import tqdm

class NIDS(BasePreprocessingPipeline):

    def __init__(self, csv_filename=None, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.base_folder = os.path.join('datasets', 'NIDS')
        self.base_name = 'NIDS'
        self.csv_filename = csv_filename
        self.name = f'{self.base_name}_{Path(csv_filename).stem}_{self.kind}'
        self.folder = self.base_folder
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        log_print(f'Converting file \'{self.csv_filename}\' to parquet.')
        df = pd.read_csv(self.csv_filename, low_memory=False)
        df = df.replace('<!DOCTYPE html>', '', regex=True)

        assert 'Label' in df.columns and 'Attack' in df.columns

        if self.binarize:
            df[self.target] = np.where(df['Attack'] == 'Benign', 'Benign', 'Malign')
        else:
            df[self.target] = df['Attack']

        df = df.drop(columns=[
            'Flow ID', 'Src IP', 'Dst IP', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
            'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Timestamp', 'Attack', 'Label', 'Dataset'
        ], errors='ignore')

        self.data = df

        parquet_filename = str(self.csv_filename).replace('.csv', f'_{self.kind}.parquet')
        df.to_parquet(parquet_filename)
        log_print(f'Converted file \'{self.csv_filename}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        parquet_filename = Path(str(self.csv_filename).replace('.csv', f'_{self.kind}.parquet'))
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

# PYTHONPATH=. python modules/preprocessing/custom/nids.py
if __name__ == "__main__":

    csv_filenames_data = [
        Path('datasets/NIDS/source/CIC-BoT-IoT/a27809afa6caa7e0_MOHANAD_A4706/data/CIC-BoT-IoT.csv'),
        Path('datasets/NIDS/source/CIC-ToN-IoT/a40a412453292fe6_MOHANAD_A4706/data/CIC-ToN-IoT.csv'),
        Path('datasets/NIDS/source/NF-BoT-IoT-v2/befb58edf3428167_MOHANAD_A4706/data/NF-BoT-IoT-v2.csv'),
        Path('datasets/NIDS/source/NF-BoT-IoT-v3/d509c9db7490cf92_NFV3DATA-A11964_A11964/data/NF-BoT-IoT-v3.csv'),
        Path('datasets/NIDS/source/NF-BoT-IoT/de2c6f75dd50d933_MOHANAD_A4706/data/NF-BoT-IoT.csv'),
        Path('datasets/NIDS/source/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv'),
        Path('datasets/NIDS/source/NF-CSE-CIC-IDS2018-v2/b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv'),
        Path('datasets/NIDS/source/NF-CSE-CIC-IDS2018/88a47ba2ab64258e_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018.csv'),
        Path('datasets/NIDS/source/NF-ToN-IoT-v2/9bafce9d380588c2_MOHANAD_A4706/data/NF-ToN-IoT-v2.csv'),
        Path('datasets/NIDS/source/NF-ToN-IoT-v3/02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv'),
        Path('datasets/NIDS/source/NF-ToN-IoT/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'),
        Path('datasets/NIDS/source/NF-UNSW-NB15-v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv'),
        Path('datasets/NIDS/source/NF-UNSW-NB15-v3/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv'),
        Path('datasets/NIDS/source/NF-UQ-NIDS-v2/9810e03bba4983da_MOHANAD_A4706/data/NF-UQ-NIDS-v2.csv'),
        Path('datasets/NIDS/source/NF-UQ-NIDS/e3bd3035f88e55fa_MOHANAD_A4706/data/NF-UQ-NIDS.csv'),
        Path('datasets/NIDS/source/NF-USNW-NB15/88695f0f620eb568_MOHANAD_A4706/data/NF-UNSW-NB15.csv')
    ]

    for binarize in tqdm([True, False], desc="Binarize", leave=False):

        for csv_filename in tqdm(csv_filenames_data, desc="NIDS_CSV", leave=False):

            try:

                log_print(f'[NIDS] Creating custom pipeline: {binarize} => {csv_filename}')

                nids = NIDS(csv_filename=csv_filename, binarize=binarize)

                nids.pipeline(preload=False, complexity_mode=None)

                log_print(f'[NIDS] Finished custom pipeline: {binarize} => {csv_filename}\n\n')

            except Exception as e:

                log_print(f'[NIDS] Error: {str(e)}')
