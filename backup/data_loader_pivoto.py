import logging
import os
import sys

import colorlog
import numpy as np
import pandas as pd
from data_utils import _drop_duplicates, _persist_dataset, _pprint_value_counts
from fastai.tabular.all import df_shrink

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s")
handler_stdout = logging.StreamHandler(stream=sys.stdout)
handler_stdout.setFormatter(formatter)
handler_file = logging.FileHandler(f"data_loader.log", mode="w")
handler_file.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler_stdout)
logger.addHandler(handler_file)


def load_iot_network_intrusion(mode, rows_limit=None):
    work_folder = os.path.join(os.path.dirname(
        __file__), f'../datasets/iot_network_intrusion_pivoto/source/')
    persist_folder = work_folder.replace('source', f'generated/{mode}')
    base_filename = f'iot_network_intrusion_{mode}.csv'
    full_filename = os.path.join(work_folder, base_filename)

    logger.info(f'Started processing file \'{full_filename}\'.')

    # for capture files processing, refer to https://github.com/marcelovca90/iot-nid-pandas
    df = pd.read_csv(filepath_or_buffer=full_filename,
                     header=0, nrows=rows_limit, low_memory=False)

    logger.info(f'Finished processing file \'{base_filename}\'.')

    cols_with_na = list(pd.isnull(df).sum()[pd.isnull(df).sum() > 0].index)

    df[cols_with_na] = df[cols_with_na].fillna(0)

    df = df.infer_objects().astype({
        '_ws.col.Time': np.int64,
        'frame.len': np.int64,
        'frame.number': np.int64,
        'icmp.code': np.uint8,
        'icmp.length': np.uint8,
        'icmp.type': np.uint8,
        'ip.proto': np.uint8,
        'ip.len': np.uint8,
        'ip.src': 'string',
        'ip.dst': 'string',
        'ip.ttl': np.uint8,
        'ip.flags.df': np.uint8,
        'ip.flags.mf': np.uint8,
        'ip.flags.rb': np.uint8,
        'ip.flags.sf': np.uint8,
        'ip.version': np.uint8,
        'tcp.flags.ack': np.uint8,
        'tcp.flags.cwr': np.uint8,
        'tcp.flags.ecn': np.uint8,
        'tcp.flags.fin': np.uint8,
        'tcp.flags.ns': np.uint8,
        'tcp.flags.push': np.uint8,
        'tcp.flags.res': np.uint8,
        'tcp.flags.reset': np.uint8,
        'tcp.flags.syn': np.uint8,
        'tcp.flags.urg': np.uint8,
        'udp.length': np.uint8,
        'udp.srcport': np.uint8,
        'udp.dstport': np.uint8,
        'label': 'category'
    })

    df = df_shrink(df=df, skip=[], obj2cat=True, int2uint=False)

    df = _drop_duplicates(df)

    logger.info('Dataframe info:')
    df.info()

    _pprint_value_counts(df, 'label')

    _persist_dataset(df, persist_folder,
                     f'iot_network_intrusion_{mode}', ['csv', 'parquet'])


if __name__ == "__main__":
    load_iot_network_intrusion('macro', None)
    load_iot_network_intrusion('micro', None)
