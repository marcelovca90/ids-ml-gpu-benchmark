import json
import os
import re
import subprocess

import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline


class IoT_Network_Intrusion_Micro(BasePreprocessingPipeline):

    @function_call_logger
    def __init__(self) -> None:
        super().__init__()
        self.folder = os.path.join('datasets', 'iot_network_intrusion')
        self.name = 'IoT_Network_Intrusion_Micro'
        self.target = 'label'

    @function_call_logger
    def run_tshark(self, id, filename, category, subcategory, filter):
        input_directory = os.path.join(os.getcwd(), self.folder, 'source')
        output_directory = os.path.join(input_directory, 'csv')
        output_filename = filename.replace(
            ".pcap", f"_{category}_{subcategory}.csv")
        output_filename = re.sub(r"[^a-zA-Z0-9._-]+", "", output_filename)
        output_filename = os.path.join(output_directory, output_filename)
        tshark_args = \
            [
                "tshark",
                "-t", "ud",
                "-r", os.path.join(input_directory, filename),
                "-Y", filter,
                "-T", "fields",
                "-E", "separator=,",
                "-E", "header=y",
                '-e', '_ws.col.Time',
                '-e', 'frame.len',
                '-e', 'frame.number',
                '-e', 'icmp.code',
                '-e', 'icmp.length',
                '-e', 'icmp.type',
                '-e', 'ip.proto',
                '-e', 'ip.len',
                '-e', 'ip.src',
                '-e', 'ip.dst',
                '-e', 'ip.ttl',
                '-e', 'ip.flags.df',
                '-e', 'ip.flags.mf',
                '-e', 'ip.flags.rb',
                '-e', 'ip.flags.sf',
                '-e', 'ip.version',
                '-e', 'tcp.flags.ack',
                '-e', 'tcp.flags.cwr',
                '-e', 'tcp.flags.ecn',
                '-e', 'tcp.flags.fin',
                '-e', 'tcp.flags.ns',
                '-e', 'tcp.flags.push',
                '-e', 'tcp.flags.res',
                '-e', 'tcp.flags.reset',
                '-e', 'tcp.flags.syn',
                '-e', 'tcp.flags.urg',
                '-e', 'udp.length',
                '-e', 'udp.srcport',
                '-e', 'udp.dstport'
            ]
        if id == 1:
            tshark_args.remove("-Y")
            tshark_args.remove("null")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        with open(output_filename, "w") as handle:
            subprocess.run(args=tshark_args, stdout=handle, check=True)

    @function_call_logger
    def aggregate_csvs(self, mode='micro'):
        df_list = []
        work_folder = os.path.join(os.getcwd(), self.folder, 'source', 'csv')
        for csv_filename in os.listdir(work_folder):
            parts = csv_filename.split('_')
            category, subcategory = parts[1], parts[2].replace('.csv', '')
            df = pd.read_csv(
                os.path.join(work_folder, csv_filename), on_bad_lines="skip")
            df['label'] = category if mode == 'macro' else f"{category}_{subcategory}"
            df_list.append(df)
        ans = pd.concat(df_list).infer_objects()
        return ans

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')

        metadata_filename = os.path.join(
            work_folder, 'dataset_description_new.json')
        with open(metadata_filename, 'r') as handle:
            metadata = json.loads(handle.read())

        for rec in metadata:
            id, filename, category, subcategories = rec['id'], rec[
                'filename'], rec['category'], rec['subcategories']
            log_print(
                f'Started processing capture with id={id} ({filename}).')
            benign_filter = "null" if id == 1 else " and ".join(
                [f"!({s['filter']})" for s in subcategories])
            self.run_tshark(id, filename, "Normal", "Normal", benign_filter)
            for subcategory in subcategories:
                malign_filter = subcategory["filter"]
                self.run_tshark(id, filename, category,
                                subcategory["name"], malign_filter)
            log_print(
                f'Finished processing capture with id={id} ({filename}).')

        df = self.aggregate_csvs()

        filename_parquet = os.path.join(work_folder, self.name + '.parquet')
        log_print(f'Saving file \'{filename_parquet}\' to parquet.')
        df.to_csv(filename_parquet, index=False)
        df.to_parquet(filename_parquet)
        log_print(f'Saving file \'{filename_parquet}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        full_filename = os.path.join(work_folder, self.name + '.parquet')
        log_print(f'Loading parquet file \'{full_filename}\'.')
        self.data = pd.read_parquet(full_filename)
        log_print(f'Loaded parquet file \'{full_filename}\'.')
