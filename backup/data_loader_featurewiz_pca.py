import logging
import os
import re
import sys

import colorlog
import numpy as np
import pandas as pd
import swifter
from data_utils import (_baseline_evaluation, _convert_to_int,
                        _downcast_dtypes, _drop_duplicates,
                        _filter_by_frequency, _label_encode, _min_max_scale,
                        _one_hot_encode, _persist_dataset, _persist_mappings,
                        _persist_subsets, _plot_feature_importances,
                        _pprint_value_counts, _replace_values,
                        _select_relevant_features, _sort_columns,
                        _train_test_split)

# from sklearnex import patch_sklearn

# patch_sklearn(global_patch=True)

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


def load_bot_iot(mode, rows_limit=None, persist_formats=['csv'], return_X_y=True):
    work_folder = os.path.join(os.path.dirname(
        __file__), '../datasets/bot_iot/source/Entire Dataset/')
    persist_folder = work_folder.replace(
        'source/Entire Dataset', f'generated/{mode}/featurewiz_pca')

    if rows_limit:
        logging.info(
            f'rows_limit = {rows_limit}; listing CSV files for BoT-IoT ({mode})...')
        data_files = list(filter(lambda f: re.match(
            r'UNSW_2018_IoT_Botnet_Dataset_\d+.csv', f), os.listdir(work_folder)))
    else:
        logging.info(
            f'rows_limit = {rows_limit}; listing FTR files for BoT-IoT ({mode})...')
        data_files = list(
            filter(lambda f: f.endswith('.ftr'), os.listdir(work_folder)))

    data_frames = []

    columns_csv_filename = os.path.join(
        work_folder, 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv')
    columns = [c.strip() for c in pd.read_csv(
        filepath_or_buffer=columns_csv_filename).columns]

    checkpoint_filename = os.path.join(persist_folder, 'checkpoint.ftr')
    if os.path.isfile(checkpoint_filename):
        logging.info(
            f'Checkpoint file present; loading intermediate file for BoT-IoT ({mode})...')
        df_c = pd.read_feather(checkpoint_filename)
    else:
        logging.info(
            f'Checkpoint file not present; loading original file(s) for BoT-IoT ({mode})...')
        for base_filename in data_files:

            full_filename = os.path.join(work_folder, base_filename)

            logger.info(f'Started processing file \'{base_filename}\'.')

            if rows_limit:
                df = pd.read_csv(filepath_or_buffer=full_filename,
                                 names=columns, nrows=rows_limit, low_memory=False)
            else:
                df = pd.read_feather(path=full_filename, columns=columns)

            if mode == 'micro':
                df['label'] = df['category'] + "_" + df['subcategory']
            elif mode == 'macro':
                df['label'] = df['category']

            # df = df.drop(columns=[
            #     'pkSeqID', 'stime', 'saddr', 'daddr', 'seq', 'ltime', 'mean', 'stddev', 'min', 'max', 'sum',
            #     'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'attack', 'category', 'subcategory'])

            # df = df.drop(columns=['pkSeqID', 'seq', 'stime', 'ltime', 'saddr', 'daddr', 'attack', 'category', 'subcategory'])
            df = df.drop(columns=['pkSeqID', 'seq', 'saddr',
                         'daddr', 'attack', 'category', 'subcategory'])

            _drop_duplicates(df, inplace=True)

            data_frames.append(df)

            logger.info(f'Finished processing file \'{base_filename}\'.')

        df_c = pd.concat(data_frames, copy=False).round(3).infer_objects()

        _pprint_value_counts(df_c, 'label')

        cols_with_na = list(pd.isnull(df_c).sum()[
                            pd.isnull(df_c).sum() > 0].index)
        df_c[cols_with_na] = df_c[cols_with_na].fillna(0)

        df_c['dport'] = df_c['dport'].astype(
            'str').swifter.apply(lambda x: _convert_to_int(x))
        df_c['sport'] = df_c['sport'].astype(
            'str').swifter.apply(lambda x: _convert_to_int(x))

        _downcast_dtypes(df_c, inplace=True)

        _min_max_scale(df_c, 'stime', inplace=True)
        _min_max_scale(df_c, 'ltime', inplace=True)

        df_c = _one_hot_encode(df_c, 'flgs')
        df_c = _one_hot_encode(df_c, 'proto')
        df_c = _one_hot_encode(df_c, 'state')

        df_c, mappings = _label_encode(df_c, 'label')

        df_c = _sort_columns(df_c, ['label'])

        _drop_duplicates(df_c, inplace=True)

        # _filter_by_frequency(df_c, 'label', 0.001, inplace=True)

        # df_c = _drop_less_relevant_columns(df_c, 'label')

        logging.info(f'Persisting checkpoint to {checkpoint_filename}...')
        df_c.reset_index(drop=False).to_feather(path=checkpoint_filename)

    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', persist_folder)
    df_c.info()

    df_c = _select_relevant_features(
        df_c, 'label', 'featurewiz', skip_sulov=False)

    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', persist_folder)
    df_c.info()

    df_c = _select_relevant_features(
        df_c, 'label', 'pca', variance_threshold='mle')

    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', persist_folder)
    df_c.info()

    _pprint_value_counts(df_c, 'label')

    X_train, X_test, y_train, y_test = _train_test_split(
        df_c, 'label', 0.2, ['tomek'])

    if persist_formats:
        _persist_mappings(mappings, persist_folder)
        _persist_dataset(df_c, persist_folder,
                         f'bot_iot_{mode}', persist_formats)
        _persist_subsets(X_train, X_test, y_train, y_test,
                         persist_folder, persist_formats)

    if return_X_y:
        return X_train, X_test, y_train, y_test
    else:
        return df_c


def load_iot_23(rows_limit=None, persist=True, return_X_y=True):
    work_folder = os.path.join(os.path.dirname(
        __file__), '../datasets/iot_23/source/')
    persist_folder = work_folder.replace('source', 'generated')
    sub_folders = filter(lambda x: ('bro' in x), [
                         x[0] for x in os.walk(work_folder)])
    base_filename = 'conn.log.labeled'
    data_frames = []

    for folder in sub_folders:

        full_filename = os.path.join(folder, base_filename)

        logger.info(f'Started processing folder \'{folder}\'.')

        df = pd.read_table(filepath_or_buffer=full_filename,
                           skiprows=8, nrows=rows_limit, low_memory=False)

        df.columns = [
            'ts',
            'uid',
            'id.orig_h',
            'id.orig_p',
            'id.resp_h',
            'id.resp_p',
            'proto',
            'service',
            'duration',
            'orig_bytes',
            'resp_bytes',
            'conn_state',
            'local_orig',
            'local_resp',
            'missed_bytes',
            'history',
            'orig_pkts',
            'orig_ip_bytes',
            'resp_pkts',
            'resp_ip_bytes',
            'label'
        ]

        df = df.drop(columns=['ts', 'uid', 'service', 'local_orig',
                     'local_resp', 'history', 'id.orig_h', 'id.resp_h'])

        df = df.drop(df.tail(1).index)

        df = _drop_duplicates(df)

        data_frames.append(df)

        logger.info(f'Finished processing folder \'{folder}\'.')

    df_c = pd.concat(data_frames, copy=False).round(3).infer_objects()

    _pprint_value_counts(df_c, 'label')

    _replace_values(df_c, 'duration',   '-',
                    np.float64(0.0))
    _replace_values(df_c, 'orig_bytes', '-',
                    np.uint64(0))
    _replace_values(df_c, 'resp_bytes', '-',
                    np.uint64(0))
    _replace_values(df_c, 'label',      '-   Malicious   Attack',
                    'Attack')
    _replace_values(df_c, 'label',      '(empty)   Malicious   Attack',
                    'Attack')
    _replace_values(df_c, 'label',      '(empty)   Benign   -',
                    'Benign')
    _replace_values(df_c, 'label',      '-   benign   -',
                    'Benign')
    _replace_values(df_c, 'label',      '-   Benign   -',
                    'Benign')
    _replace_values(df_c, 'label',      '(empty)   Benign   -',
                    'Benign')
    _replace_values(df_c, 'label',      'CARhxZ3hLNVO3xYFok   Benign   -',
                    'Benign')
    _replace_values(df_c, 'label',      'COLnd035cNITygYHp3   Benign   -',
                    'Benign')
    _replace_values(df_c, 'label',      '-   Malicious   C&C',
                    'C&C')
    _replace_values(df_c, 'label',      '(empty)   Malicious   C&C',
                    'C&C')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-FileDownload',
                    'C&C-FileDownload')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat',
                    'C&C-HeartBeat')
    _replace_values(df_c, 'label',      '(empty)   Malicious   C&C-HeartBeat',
                    'C&C-HeartBeat')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat-Attack',
                    'C&C-HeartBeat-Attack')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat-FileDownload',
                    'C&C-HeartBeat-FileDownload')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat-PartOfAHorizontalPortScan',
                    'C&C-HeartBeat-PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-Mirai',
                    'C&C-Mirai')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-PartOfAHorizontalPortScan',
                    'C&C-PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-Torii',
                    'C&C-Torii')
    _replace_values(df_c, 'label',      '-   Malicious   DDoS',
                    'DDoS')
    _replace_values(df_c, 'label',      '(empty)   Malicious   DDoS',
                    'DDoS')
    _replace_values(df_c, 'label',      '-   Malicious   FileDownload',
                    'FileDownload')
    _replace_values(df_c, 'label',      '-   Malicious   Okiru',
                    'Okiru')
    _replace_values(df_c, 'label',      '(empty)   Malicious   Okiru',
                    'Okiru')
    _replace_values(df_c, 'label',      '-   Malicious   Okiru-Attack',
                    'Okiru-Attack')
    _replace_values(df_c, 'label',      '-   Malicious   PartOfAHorizontalPortScan',
                    'PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '(empty)   Malicious   PartOfAHorizontalPortScan',
                    'PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '-   Malicious   PartOfAHorizontalPortScan-Attack',
                    'PartOfAHorizontalPortScan-Attack')

    _pprint_value_counts(df_c, 'label')

    df_c = df_c.infer_objects().astype({
        'id.orig_p': np.uint64,
        'id.resp_p': np.uint64,
        'duration': np.float64,
        'orig_bytes': np.uint64,
        'resp_bytes': np.uint64,
        'missed_bytes': np.uint64,
        'orig_pkts': np.uint64,
        'orig_ip_bytes': np.uint64,
        'resp_pkts': np.uint64,
        'resp_ip_bytes': np.uint64,
        'label': 'category'
    })

    df_c = _drop_duplicates(df_c)

    # df_c = _filter_by_quantile(df_c, 'label')

    df_c = _filter_by_frequency(df_c, 'label', 0.001)

    df_c = _one_hot_encode(df_c, 'proto')

    df_c, _ = _label_encode(df_c, 'conn_state')

    df_c, mappings = _label_encode(df_c, 'label')

    # df_c = _drop_less_relevant_columns(df_c, 'label')

    df_c = _sort_columns(df_c, ['label'])

    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', persist_folder)
    df_c.info()

    df_c = _select_relevant_features(df_c, 'label', 10)

    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', persist_folder)
    df_c.info()

    X_train, X_test, y_train, y_test = _train_test_split(
        df_c, 'label', 0.2, True, 10)

    _pprint_value_counts(df_c, 'label')

    if persist:
        _persist_mappings(mappings, persist_folder)
        _persist_dataset(df_c, persist_folder, 'iot_23')
        _persist_subsets(X_train, X_test, y_train, y_test, persist_folder)

    if return_X_y:
        return X_train, X_test, y_train, y_test
    else:
        return df_c


def load_mqtt_iot_ids2020(rows_limit=None, persist=True, return_X_y=True):
    work_folder = os.path.join(os.path.dirname(
        __file__), '../datasets/mqtt_iot_ids2020/source/')
    persist_folder = work_folder.replace('source', 'generated')
    csv_files = list(filter(lambda f: f.endswith(
        '.csv'), os.listdir(work_folder)))
    data_frames = []

    for base_filename in csv_files:

        full_filename = os.path.join(work_folder, base_filename)

        logger.info(f'Started processing file \'{base_filename}\'.')

        df = pd.read_csv(filepath_or_buffer=full_filename,
                         header=0, nrows=rows_limit, low_memory=False)

        df.columns = [
            'timestamp',
            'src_ip',
            'dst_ip',
            'protocol',
            'ttl',
            'ip_len',
            'ip_flag_df',
            'ip_flag_mf',
            'ip_flag_rb',
            'src_port',
            'dst_port',
            'tcp_flag_res',
            'tcp_flag_ns',
            'tcp_flag_cwr',
            'tcp_flag_ecn',
            'tcp_flag_urg',
            'tcp_flag_ack',
            'tcp_flag_push',
            'tcp_flag_reset',
            'tcp_flag_syn',
            'tcp_flag_fin',
            'mqtt_messagetype',
            'mqtt_messagelength',
            'mqtt_flag_uname',
            'mqtt_flag_passwd',
            'mqtt_flag_retain',
            'mqtt_flag_qos',
            'mqtt_flag_willflag',
            'mqtt_flag_clean',
            'mqtt_flag_reserved',
            'is_attack'
        ]

        df = df.drop(columns=['timestamp', 'src_ip', 'dst_ip'])

        df = _drop_duplicates(df)

        df = df[df['is_attack'] != 'is_attack']

        _replace_values(df, 'is_attack',   0, 'normal')
        _replace_values(df, 'is_attack', '0', 'normal')
        _replace_values(df, 'is_attack',   1,
                        base_filename.replace('.csv', ''))
        _replace_values(df, 'is_attack', '1',
                        base_filename.replace('.csv', ''))

        data_frames.append(df)

        logger.info(f'Finished processing file \'{base_filename}\'.')

    df_c = pd.concat(data_frames, copy=False).round(3).infer_objects()

    _pprint_value_counts(df_c, 'is_attack')

    cols_with_na = list(pd.isnull(df_c).sum()[pd.isnull(df_c).sum() > 0].index)
    df_c[cols_with_na] = df_c[cols_with_na].fillna(0)

    df_c = df_c.infer_objects().astype({
        'ttl': np.uint8,
        'ip_len': np.int64,
        'ip_flag_df': np.uint8,
        'ip_flag_mf': np.uint8,
        'ip_flag_rb': np.uint8,
        'src_port': np.int64,
        'dst_port': np.int64,
        'tcp_flag_res': np.uint8,
        'tcp_flag_ns': np.uint8,
        'tcp_flag_cwr': np.uint8,
        'tcp_flag_ecn': np.uint8,
        'tcp_flag_urg': np.uint8,
        'tcp_flag_ack': np.uint8,
        'tcp_flag_push': np.uint8,
        'tcp_flag_reset': np.uint8,
        'tcp_flag_syn': np.uint8,
        'tcp_flag_fin': np.uint8,
        'mqtt_messagetype': np.uint8,
        'mqtt_messagelength': np.uint8,
        'mqtt_flag_uname': np.uint8,
        'mqtt_flag_passwd': np.uint8,
        'mqtt_flag_retain': np.uint8,
        'mqtt_flag_qos': np.uint8,
        'mqtt_flag_willflag': np.uint8,
        'mqtt_flag_clean': np.uint8,
        'mqtt_flag_reserved': np.uint8,
        'is_attack': 'category'
    })

    df_c = _drop_duplicates(df_c)

    df_c = _filter_by_frequency(df_c, 'is_attack', 0.001)

    df_c = _one_hot_encode(df_c, 'protocol')

    df_c, mappings = _label_encode(df_c, 'is_attack')

    df_c = _sort_columns(df_c, ['is_attack'])

    _baseline_evaluation(df_c, 'is_attack')
    _plot_feature_importances(df_c, 'is_attack', persist_folder,)
    df_c.info()

    df_c = _select_relevant_features(df_c, 'is_attack', 10)

    _baseline_evaluation(df_c, 'is_attack')
    _plot_feature_importances(df_c, 'is_attack', persist_folder)
    df_c.info()

    X_train, X_test, y_train, y_test = _train_test_split(
        df_c, 'is_attack', 0.2, True, 10)

    _pprint_value_counts(df_c, 'is_attack')

    if persist:
        _persist_mappings(mappings, persist_folder)
        _persist_dataset(df_c, persist_folder, 'mqtt_iot_ids2020')
        _persist_subsets(X_train, X_test, y_train, y_test, persist_folder)

    if return_X_y:
        return X_train, X_test, y_train, y_test
    else:
        return df_c


def load_iot_network_intrusion(mode, rows_limit=None, persist=True, return_X_y=True):
    work_folder = os.path.join(os.path.dirname(
        __file__), f'../datasets/iot_network_intrusion/source/{mode}/')
    persist_folder = work_folder.replace('source', 'generated')
    base_filename = f'iot_network_intrusion.csv'
    full_filename = os.path.join(work_folder, base_filename)

    logger.info(f'Started processing file \'{full_filename}\'.')

    # for capture files processing, refer to https://github.com/marcelovca90/iot-nid-pandas
    df_c = pd.read_csv(filepath_or_buffer=full_filename,
                       header=0, nrows=rows_limit, low_memory=False)

    logger.info(f'Finished processing file \'{base_filename}\'.')

    df_c = df_c.drop(columns=['ip.src', 'ip.dst'])

    _pprint_value_counts(df_c, 'label')

    cols_with_na = list(pd.isnull(df_c).sum()[pd.isnull(df_c).sum() > 0].index)
    df_c[cols_with_na] = df_c[cols_with_na].fillna(0)

    df_c = df_c.infer_objects().astype({
        '_ws.col.Time': np.int64,
        'frame.len': np.int64,
        'frame.number': np.int64,
        'icmp.code': np.uint8,
        'icmp.length': np.uint8,
        'icmp.type': np.uint8,
        'ip.proto': np.uint8,
        'ip.len': np.uint8,
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

    df_c = _drop_duplicates(df_c)

    df_c = _filter_by_frequency(df_c, 'label', 0.001)

    # df_c = _one_hot_encode(df_c, 'ip.proto')

    df_c, mappings = _label_encode(df_c, 'label')

    df_c = _sort_columns(df_c, ['label'])

    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', persist_folder)
    df_c.info()

    df_c = _select_relevant_features(df_c, 'label', 10)

    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', persist_folder)
    df_c.info()

    X_train, X_test, y_train, y_test = _train_test_split(
        df_c, 'label', 0.2, True, 10)

    _pprint_value_counts(df_c, 'label')

    if persist:
        _persist_mappings(mappings, persist_folder)
        _persist_dataset(df_c, persist_folder, 'iot_network_intrusion')
        _persist_subsets(X_train, X_test, y_train, y_test, persist_folder)

    if return_X_y:
        return X_train, X_test, y_train, y_test
    else:
        return df_c


if __name__ == "__main__":
    logger.info('#### FEATUREWIZ_PCA_MACRO ####')
    load_bot_iot('macro', None, ['ftr'], False)
    logger.info('#### FEATUREWIZ_PCA_MICRO ####')
    load_bot_iot('micro', None, ['ftr'], False)
