import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline

sys.path.append(Path(__file__).absolute().parent.parent)

class CICIoMT2024_Bluetooth(BasePreprocessingPipeline):

    @function_call_logger
    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'CICIoMT2024')
        self.name = 'CICIoMT2024_Bluetooth'
        self.target = 'label'

    @function_call_logger
    def run_tshark(self, input_filename):
        output_filename = input_filename.replace('pcap', 'csv')
        os.makedirs(Path(output_filename).parent, exist_ok=True)
        tshark_args = [
            "tshark",
            "-t", "ud",
            "-r", input_filename,
            "-Y", "bthci_evt || bthci_cmd || btl2cap || btatt || btrfcomm",
            "-T", "fields",
            "-E", "separator=,",
            "-E", "header=y",
            "-e", "_ws.col.Time",
            "-e", "frame.len",
            "-e", "frame.number",
            "-e", "bthci_cmd.opcode",
            "-e", "btl2cap.cid",
            "-e", "btl2cap.length",
            "-e", "btrfcomm.channel",
            "-e", "btatt.opcode",
            "-e", "btatt.handle",
            "-e", "btatt.value",
        ]
        with open(output_filename, "w") as handle:
            try:
                subprocess.run(
                    args=tshark_args, check=True, text=True,
                    stdout=handle, stderr=subprocess.PIPE
                )
            except Exception as e:
                print('Error:', str(e))

    @function_call_logger
    def aggregate_csvs(self):
        df_list = []
        work_folder = os.path.join(os.getcwd(), self.folder, 'source/Bluetooth/attacks/csv')
        csv_files = glob.glob(os.path.join(work_folder, '**', '*.csv'), recursive=True)
        for csv_filename in csv_files:
            category = Path(csv_filename).stem.split('_')[1]
            df = pd.read_csv(csv_filename, low_memory=False, on_bad_lines="skip")
            df = df.drop(columns=['_ws.col.Time', '_ws.col.cls_time', 'frame.number'], errors='ignore')
            df['label'] = category

            def hex_to_int(val):
                if isinstance(val, str) and val.startswith('0x'):
                    try:
                        return int(val, 16)
                    except ValueError:
                        return val
                return val

            for col in df.columns:
                # Only apply if the column has at least one '0x'-prefixed string
                if df[col].astype(str).str.startswith('0x').any():
                    df[col] = df[col].apply(hex_to_int)

            df_list.append(df)

        ans = pd.concat(df_list).infer_objects()

        if self.binarize:
            ans['label'] = np.where(
                (ans['label'] == 'Benign'), 'Benign', 'Malign'
            )

        return ans

    @function_call_logger
    def prepare(self) -> None:
        input_filenames = [
            os.path.join(os.getcwd(), self.folder, 'source/Bluetooth/attacks/pcap/train/Bluetooth_Benign_train.pcap'),
            os.path.join(os.getcwd(), self.folder, 'source/Bluetooth/attacks/pcap/train/Bluetooth_DoS_train.pcap'),
            os.path.join(os.getcwd(), self.folder, 'source/Bluetooth/attacks/pcap/test/Bluetooth_Benign_test.pcap'),
            os.path.join(os.getcwd(), self.folder, 'source/Bluetooth/attacks/pcap/test/Bluetooth_DoS_test.pcap'),
        ]

        for filename in input_filenames:
            log_print(f'Started processing capture \'{filename}\'.')
            self.run_tshark(filename)
            log_print(f'Finished processing capture \'{filename}\'.')

        df = self.aggregate_csvs()

        filename_parquet = os.path.join(os.getcwd(), self.folder, f'source/Bluetooth/attacks/parquet/{self.name}.parquet')
        os.makedirs(Path(filename_parquet).parent, exist_ok=True)
        log_print(f'Saving file \'{filename_parquet}\' to parquet.')
        df.to_parquet(filename_parquet)
        log_print(f'Saved  file \'{filename_parquet}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source/Bluetooth/attacks/parquet')
        full_filename = os.path.join(work_folder, f'{self.name}.parquet')
        log_print(f'Loading parquet file \'{full_filename}\'.')
        self.data = pd.read_parquet(full_filename)
        log_print(f'Loaded parquet file \'{full_filename}\'.')
