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

class EDGE_IIOTSET(BasePreprocessingPipeline):

    def __init__(self, csv_filename=None, binarize=False) -> None:
        super().__init__(binarize=binarize)
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

    binarize_flags = [False]

    csv_filenames = [
        "ML-EdgeIIoT-dataset.csv",
        "DNN-EdgeIIoT-dataset.csv"
    ]

    for i, binarize_flag in enumerate(tqdm(binarize_flags, desc="Binarize", leave=False)):

        for j, csv_filename in enumerate(tqdm(csv_filenames, desc="EDGE_IIOTSET_CSV", leave=False)):

            try:

                msg_prefix = f"[{i+1:02}/{len(binarize_flags):02}] [{j+1:02}/{len(csv_filenames):02}]"

                log_print(f'{msg_prefix} Started processing EDGE_IIOTSET/{csv_filename} (binarize={binarize_flag}).')
                post_disc(f'{msg_prefix} Started processing EDGE_IIOTSET/{csv_filename} (binarize={binarize_flag}).')

                subfolder_path = os.path.join('datasets/EDGE_IIOTSET/source', csv_filename)

                edge_iiotset = EDGE_IIOTSET(csv_filename=csv_filename, binarize=binarize_flag)

                edge_iiotset.pipeline()

                log_print(f'{msg_prefix} Finished processing EDGE_IIOTSET/{csv_filename} (binarize={binarize_flag}).')
                post_disc(f'{msg_prefix} Finished processing EDGE_IIOTSET/{csv_filename} (binarize={binarize_flag}).')

            except Exception as e:

                log_print(f'{msg_prefix} Error processing EDGE_IIOTSET/{csv_filename} (binarize={binarize_flag}): {str(e)}')
                post_disc(f'{msg_prefix} Error processing EDGE_IIOTSET/{csv_filename} (binarize={binarize_flag}): {str(e)}')
