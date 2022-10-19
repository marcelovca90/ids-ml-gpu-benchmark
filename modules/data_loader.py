import json
import logging
import os
import sys
from calendar import c
from collections import OrderedDict
from operator import getitem

import colorlog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.linear_model import ElasticNet, Lasso, LassoLars, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import ExtraTreeClassifier

SEED = 10

formatter = colorlog.ColoredFormatter("%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s")
handler_stdout = logging.StreamHandler(stream=sys.stdout)
handler_stdout.setFormatter(formatter)
handler_file = logging.FileHandler(f"data_loader.log", mode="w")
handler_file.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler_stdout)
logger.addHandler(handler_file)

def _baseline_evaluation(df, label_column='label'):
    X_train, X_test, y_train, y_test = _train_test_split(df, label_column)
    cls = ExtraTreeClassifier(random_state=SEED)
    cls.fit(X_train, y_train)
    logger.info(f"Shape: {df.shape} => Training Score: {cls.score(X_train, y_train)}")
    logger.info(f"Shape: {df.shape} => Test Score    : {cls.score(X_test, y_test)}\n")

def _drop_duplicates(df, create_count_column=False):
    if create_count_column:
        current_cols = [x for x in df.columns]
        logger.info(f'Duplicated rows before removal: {df.duplicated().sum()}')
        df['count'] = 1
        df = df.groupby(current_cols)['count'].count().reset_index().drop_duplicates()
        logger.info(f'Duplicated after before removal: {df.duplicated().sum()}')
        return df
    else:
        logger.info(f'Dropping {df.duplicated().sum()} duplicated rows.')
        return df.drop_duplicates()

def _drop_less_relevant_columns(df, label_column, threshold=0):
    constant_filter = VarianceThreshold(threshold=threshold).fit(df.drop(columns=[label_column]))
    support_columns = df.drop(columns=[label_column]).columns[constant_filter.get_support()]
    non_constant_columns = [col for col in df.columns if col not in support_columns]
    logger.info(f'\nNon (quasi-)constant columns: {non_constant_columns}')
    constant_columns = [col for col in df.columns if col in support_columns]
    logger.info(f'\n(Quasi-)constant columns: {constant_columns}')
    return df.drop(columns=constant_columns)

def _filter_by_instance_hardness(X, y, n_folds=5):
    logger.info(f"\nX and y shapes before filtering by instance hardness:\n{X.shape}; {y.shape}")
    X, y = InstanceHardnessThreshold(estimator=ExtraTreeClassifier(random_state=SEED), cv=n_folds, random_state=SEED).fit_resample(X, y)
    logger.info(f"\nX and y shapes after filtering by instance hardness:\n{X.shape}; {y.shape}")
    return X, y

def _filter_by_frequency(df, column, min_rel_freq_pct=0.01):
    logger.info(f"\nValue counts before filtering by frequency:")
    _pretty_print_value_counts(df, column)
    vcd = df[column].value_counts(normalize=True).to_dict()
    relevant_labels = [key for key,value in vcd.items() if value > min_rel_freq_pct/100.0]
    logger.info(f'\nDropping rows with relative frequency inferior to {min_rel_freq_pct:.3f}% ****')
    filtered_labels = df[column].value_counts().index.drop(relevant_labels)
    for label in filtered_labels:
        df = df.drop(df[df.label == label].index)
    logger.info(f"\nValue counts after filtering by frequency:")
    _pretty_print_value_counts(df, column)
    return df

def _filter_by_quantile(df, column, percentage=0.05):
    value_counts = df[column].value_counts()
    logger.info(f"\nValue counts before filtering by quantile:")
    _pretty_print_value_counts(df, column)
    threshold = value_counts.quantile(percentage)
    logger.info(f"\nDropping '{column}' rows with less than {threshold:.2f} occurrences.")
    df = df[df[column].isin(value_counts.index[value_counts.ge(threshold)])]
    logger.info(f"\nValue counts before filtering by quantile:\n")
    _pretty_print_value_counts(df, column)
    logger.info(f'\nFiltering by quantile performed succesfully; new DF shape: {df.shape}.')
    return df

def _one_hot_encode(df, column):
    df = pd.get_dummies(df, columns=[column])
    logger.info(f'\nColumn \'{column}\' successfully one-hot-encoded; new DF shape: {df.shape}.')
    return df

def _label_encode(df, column):
    encoder = LabelEncoder().fit(df[column])
    logger.info(f"\nLabel encoder found the following classes for '{column}':\n{encoder.classes_}")
    df[column] = encoder.transform(df[column])
    mappings = {}
    for _class in encoder.classes_:
        mappings.update({str(_class): str(encoder.transform([_class])[0])})
    return df.astype({column: np.uint8}), mappings

def _persist_mappings(mappings, base_folder, filename='mappings.json'):
    full_filename = os.path.join(base_folder, filename)
    with open(full_filename, 'w') as fp:
        json.dump(mappings, fp)

def _persist_dataset(df, base_folder, file_name, format='csv'):
    if format == 'csv':
        df.round(6).to_csv(os.path.join(base_folder, f'{file_name}.csv'), float_format='%g', header=None, index=None)
    elif format == 'hdf':
        df.round(6).to_hdf(os.path.join(base_folder, f'{file_name}.h5'), file_name)
    else:
        raise ValueError("Invalid format. It must be either 'csv' or 'hdf'.")

def _persist_subsets(X_train, X_test, y_train, y_test, base_folder, format='csv'):
    if format == 'csv':
        pd.DataFrame(X_train).round(6).to_csv(os.path.join(base_folder, f'X_train.csv'), float_format='%g', header=None, index=None)
        pd.DataFrame(X_test).round(6).to_csv(os.path.join(base_folder, f'X_test.csv'), float_format='%g', header=None, index=None)
        pd.DataFrame(y_train).round(6).to_csv(os.path.join(base_folder, f'y_train.csv'), float_format='%g', header=None, index=None)
        pd.DataFrame(y_test).round(6).to_csv(os.path.join(base_folder, f'y_test.csv'), float_format='%g', header=None, index=None)
    elif format == 'hdf':
        pd.DataFrame(X_train).round(6).to_hdf(os.path.join(base_folder, f'X_train.h5'), 'X_train')
        pd.DataFrame(X_test).round(6).to_hdf(os.path.join(base_folder, f'X_test.h5'), 'X_test',)
        pd.DataFrame(y_train).round(6).to_hdf(os.path.join(base_folder, f'y_train.h5'), 'y_train')
        pd.DataFrame(y_test).round(6).to_hdf(os.path.join(base_folder, f'y_test.h5'), 'y_test')
    else:
        raise ValueError("Invalid format. It must be either 'csv' or 'hdf'.")

def _plot_feature_importances(df, label_column, base_folder):
    # get importances from a tree-based classifier
    X, y = df.drop(columns=[label_column]), df[label_column]
    cls = ExtraTreeClassifier(random_state=SEED)
    cls.fit(X, y)
    importances = cls.feature_importances_
    indices = np.argsort(importances)
    features = X.columns
    # plot importances for each feature
    plt.clf()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(os.path.join(base_folder, f'feature_importances_{len(features)}.png'))

def _pretty_print_value_counts(df, column, lpad=64, rpad=12):
    vc_norm_f = df[column].value_counts(normalize=False).to_dict()
    vc_norm_t = df[column].value_counts(normalize=True).to_dict()
    ans = {}
    for label in df[column].unique():
        ans[label] = {'abs' : vc_norm_f[label], 'rel' : vc_norm_t[label]}        
    ans = OrderedDict(sorted(ans.items(), key=lambda x: getitem(x[1], 'abs'), reverse=True))
    output = []
    output.append(f"+-{'-----'.ljust(lpad,'-')}-+-{'-----'.rjust(rpad,'-')}-+-{'---------'.rjust(rpad,'-')}-+")
    output.append(f"| {column.center(lpad)} | {'Count'.center(rpad)} | {'Count (%)'.center(rpad)} |")
    output.append(f"+-{'-----'.ljust(lpad,'-')}-+-{'-----'.rjust(rpad,'-')}-+-{'---------'.rjust(rpad,'-')}-+")
    for key,value in ans.items():
        col_1 = f"{str(key).ljust(lpad)}"
        col_2 = f"{value['abs']}".rjust(rpad)
        col_3 = f"{(100.0 * value['rel']):.06f}".rjust(rpad)
        output.append(f'| {col_1} | {col_2} | {col_3} |')
    output.append(f"+-{'-----'.ljust(lpad,'-')}-+-{'-----'.rjust(rpad,'-')}-+-{'---------'.rjust(rpad,'-')}-+")
    for line in output:
        logger.info(line)
    return output

def _replace_values(df, column, old_value, new_value):
    df.loc[(df[column] == old_value), column] = new_value

def _select_relevant_features(df, label_column, n_folds=5):
    logger.info(f'\nPerforming {n_folds}-fold recursive feature elimination:')
    X, y = df.drop(columns=[label_column]).select_dtypes(include='number'), df[label_column]
    rfecv = RFECV(estimator=ExtraTreeClassifier(random_state=SEED), cv=n_folds, verbose=1)
    rfecv.fit(X, y)
    feature_mask = X.columns[rfecv.get_support()]
    relevant_columns = [col for col in X.columns if col not in feature_mask]
    logger.info(f'\nFeatures that will be kept: {relevant_columns}')
    irrelevant_cols = [col for col in X.columns if col in feature_mask]
    logger.info(f'\nFeatures that will be dropped: {irrelevant_cols}')
    return df.drop(columns=irrelevant_cols)

def _sort_columns(df, rightmost_columns):
    final_cols = [x for x in df.columns.values if x not in rightmost_columns]
    final_cols.extend(rightmost_columns)
    logger.info(f'\nColumns sorted according to {final_cols}.\n')
    return df.reindex(columns=final_cols)

def _train_test_split(df, label_column, test_size=0.2, filter_rows=False, n_folds=5):
    X, y = df.drop(columns=[label_column]), df[label_column]
    if filter_rows:
        X, y = _filter_by_instance_hardness(X, y, n_folds)
    return train_test_split(X, y, test_size=test_size, random_state=SEED)

def load_iot_23(rows_limit=None, persist=True, return_X_y=True):
    work_folder = os.path.join(os.path.dirname(__file__), '../datasets/iot_23/')
    sub_folders = filter(lambda x : ('bro' in x), [x[0] for x in os.walk(work_folder)])
    base_filename = 'conn.log.labeled'
    data_frames = []

    for folder in sub_folders:
        
        full_filename = os.path.join(folder, base_filename)

        logger.info(f'Started processing folder \'{folder}\'.')
        
        df = pd.read_table(filepath_or_buffer=full_filename, skiprows=8, nrows=rows_limit, low_memory=False)
        
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
        
        df.drop(columns=['ts', 'uid', 'service', 'local_orig', 'local_resp', 'history', 'id.orig_h', 'id.resp_h'], inplace=True)
        
        df.drop(df.tail(1).index, inplace=True)
        
        df = _drop_duplicates(df)

        data_frames.append(df)
        
        logger.info(f'Finished processing folder \'{folder}\'; DF shape: {df.shape}.\n')

    df_c = pd.concat(data_frames)

    logger.info(f"Value counts before replacing labels:")
    _pretty_print_value_counts(df_c, 'label')

    _replace_values(df_c, 'duration',   '-',                                                       np.float64(0.0))
    _replace_values(df_c, 'orig_bytes', '-',                                                       np.uint64(0))
    _replace_values(df_c, 'resp_bytes', '-',                                                       np.uint64(0))
    _replace_values(df_c, 'label',      '-   Malicious   Attack',                                  'Attack')
    _replace_values(df_c, 'label',      '(empty)   Malicious   Attack',                            'Attack')
    _replace_values(df_c, 'label',      '(empty)   Benign   -',                                    'Benign')
    _replace_values(df_c, 'label',      '-   benign   -',                                          'Benign')
    _replace_values(df_c, 'label',      '-   Benign   -',                                          'Benign')
    _replace_values(df_c, 'label',      '(empty)   Benign   -',                                    'Benign')
    _replace_values(df_c, 'label',      'CARhxZ3hLNVO3xYFok   Benign   -',                         'Benign')
    _replace_values(df_c, 'label',      'COLnd035cNITygYHp3   Benign   -',                         'Benign')
    _replace_values(df_c, 'label',      '-   Malicious   C&C',                                     'C&C')
    _replace_values(df_c, 'label',      '(empty)   Malicious   C&C',                               'C&C')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-FileDownload',                        'C&C-FileDownload')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat',                           'C&C-HeartBeat')
    _replace_values(df_c, 'label',      '(empty)   Malicious   C&C-HeartBeat',                     'C&C-HeartBeat')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat-Attack',                    'C&C-HeartBeat-Attack')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat-FileDownload',              'C&C-HeartBeat-FileDownload')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-HeartBeat-PartOfAHorizontalPortScan', 'C&C-HeartBeat-PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-Mirai',                               'C&C-Mirai')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-PartOfAHorizontalPortScan',           'C&C-PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '-   Malicious   C&C-Torii',                               'C&C-Torii')
    _replace_values(df_c, 'label',      '-   Malicious   DDoS',                                    'DDoS')
    _replace_values(df_c, 'label',      '(empty)   Malicious   DDoS',                              'DDoS')
    _replace_values(df_c, 'label',      '-   Malicious   FileDownload',                            'FileDownload')
    _replace_values(df_c, 'label',      '-   Malicious   Okiru',                                   'Okiru')
    _replace_values(df_c, 'label',      '(empty)   Malicious   Okiru',                             'Okiru')
    _replace_values(df_c, 'label',      '-   Malicious   Okiru-Attack',                            'Okiru-Attack')
    _replace_values(df_c, 'label',      '-   Malicious   PartOfAHorizontalPortScan',               'PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '(empty)   Malicious   PartOfAHorizontalPortScan',         'PartOfAHorizontalPortScan')
    _replace_values(df_c, 'label',      '-   Malicious   PartOfAHorizontalPortScan-Attack',        'PartOfAHorizontalPortScan-Attack')

    logger.info(f"\nValue counts after replacing labels:")
    _pretty_print_value_counts(df_c, 'label')

    df_c = df_c.infer_objects().astype({
        'id.orig_p'     : np.uint64,
        'id.resp_p'     : np.uint64,
        'duration'      : np.float64,
        'orig_bytes'    : np.uint64,
        'resp_bytes'    : np.uint64,
        'missed_bytes'  : np.uint64,
        'orig_pkts'     : np.uint64,
        'orig_ip_bytes' : np.uint64,
        'resp_pkts'     : np.uint64,
        'resp_ip_bytes' : np.uint64,
        'label'         : 'category'
    })

    df_c = _drop_duplicates(df_c)

    # df_c = _filter_by_quantile(df_c, 'label')

    df_c = _filter_by_frequency(df_c, 'label', 0.001) 

    df_c = _one_hot_encode(df_c, 'proto')

    df_c, _ = _label_encode(df_c, 'conn_state')

    df_c, mappings = _label_encode(df_c, 'label')

    # df_c = _drop_less_relevant_columns(df_c, 'label')

    df_c = _sort_columns(df_c, ['label'])

    logger.info(f"Performing baseline evaluation and plotting feature importances before feature selection:")
    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', work_folder)
    df_c.info()

    df_c = _select_relevant_features(df_c, 'label', 10)

    logger.info(f"\nPerforming baseline evaluation and plotting feature importances after feature selection:")
    _baseline_evaluation(df_c, 'label')
    _plot_feature_importances(df_c, 'label', work_folder)
    df_c.info()

    X_train, X_test, y_train, y_test = _train_test_split(df_c, 'label', 0.2, True, 10)

    logger.info(f"\nFinal value counts:")
    _pretty_print_value_counts(df_c, 'label')

    logger.info(f'X_train shape: {X_train.shape}; y_train shape: {y_train.shape}; y_train unique values: {set(y_train)}')
    logger.info(f'X_test shape: {X_test.shape}; y_test shape: {y_test.shape}; y_test unique values: {set(y_test)}')

    if persist:
        _persist_mappings(mappings, work_folder)
        _persist_dataset(df_c, work_folder, 'iot_23')
        _persist_subsets(X_train, X_test, y_train, y_test, work_folder)

    if return_X_y:
        return X_train, X_test, y_train, y_test
    else:
        return df_c

def load_mqtt_iot_ids2020(rows_limit=None, persist=True, return_X_y=True):
	
    work_folder = os.path.join(os.path.dirname(__file__), '../datasets/mqtt_iot_ids2020/')
    csv_files = list(filter(lambda f: f.endswith('.csv'), os.listdir(work_folder)))
    data_frames = []

    for base_filename in csv_files:
        
        full_filename = os.path.join(work_folder, base_filename)

        logger.info(f'Started processing file \'{base_filename}\'.')
        
        df = pd.read_csv(filepath_or_buffer=full_filename, header=0, nrows=rows_limit, low_memory=False)
        
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
        
        df.drop(columns=['timestamp', 'src_ip', 'dst_ip'], inplace=True)

        df = _drop_duplicates(df)

        df = df[df['is_attack'] != 'is_attack']

        _replace_values(df, 'is_attack',   0, 'normal')
        _replace_values(df, 'is_attack', '0', 'normal')
        _replace_values(df, 'is_attack',   1, base_filename.replace('.csv', ''))
        _replace_values(df, 'is_attack', '1', base_filename.replace('.csv', ''))

        data_frames.append(df)
        
        logger.info(f'Finished processing file \'{full_filename}\'; DF shape: {df.shape}.\n')

    df_c = pd.concat(data_frames)

    logger.info(f"Initial value counts:")
    _pretty_print_value_counts(df_c, 'is_attack')
    
    cols_with_na = list(pd.isnull(df_c).sum()[pd.isnull(df_c).sum() > 0].index)

    df_c[cols_with_na] = df_c[cols_with_na].fillna(0)

    df_c = df_c.infer_objects().astype({
        'ttl'                : np.uint8,
        'ip_len'             : np.int64,
        'ip_flag_df'         : np.uint8,
        'ip_flag_mf'         : np.uint8,
        'ip_flag_rb'         : np.uint8,
        'src_port'           : np.int64,
        'dst_port'           : np.int64,
        'tcp_flag_res'       : np.uint8,
        'tcp_flag_ns'        : np.uint8,
        'tcp_flag_cwr'       : np.uint8,
        'tcp_flag_ecn'       : np.uint8,
        'tcp_flag_urg'       : np.uint8,
        'tcp_flag_ack'       : np.uint8,
        'tcp_flag_push'      : np.uint8,
        'tcp_flag_reset'     : np.uint8,
        'tcp_flag_syn'       : np.uint8,
        'tcp_flag_fin'       : np.uint8,
        'mqtt_messagetype'   : np.uint8,
        'mqtt_messagelength' : np.uint8,
        'mqtt_flag_uname'    : np.uint8,
        'mqtt_flag_passwd'   : np.uint8,
        'mqtt_flag_retain'   : np.uint8,
        'mqtt_flag_qos'      : np.uint8,
        'mqtt_flag_willflag' : np.uint8,
        'mqtt_flag_clean'    : np.uint8,
        'mqtt_flag_reserved' : np.uint8,
        'is_attack'          : 'category'
    })

    df_c = _drop_duplicates(df_c)

    df_c = _filter_by_frequency(df_c, 'is_attack', 0.001) 

    df_c = _one_hot_encode(df_c, 'protocol')

    df_c, mappings = _label_encode(df_c, 'is_attack')

    df_c = _sort_columns(df_c, ['is_attack'])
    
    logger.info(f"Performing baseline evaluation and plotting feature importances before feature selection:")
    _baseline_evaluation(df_c, 'is_attack')
    _plot_feature_importances(df_c, 'is_attack', work_folder)
    df_c.info()

    df_c = _select_relevant_features(df_c, 'is_attack', 10)

    logger.info(f"\nPerforming baseline evaluation and plotting feature importances after feature selection:")
    _baseline_evaluation(df_c, 'is_attack')
    _plot_feature_importances(df_c, 'is_attack', work_folder)
    df_c.info()

    X_train, X_test, y_train, y_test = _train_test_split(df_c, 'is_attack', 0.2, True, 10)

    logger.info(f"\nFinal value counts:")
    _pretty_print_value_counts(df_c, 'is_attack')

    logger.info(f'X_train shape: {X_train.shape}; y_train shape: {y_train.shape}; y_train unique values: {set(y_train)}')
    logger.info(f'X_test shape: {X_test.shape}; y_test shape: {y_test.shape}; y_test unique values: {set(y_test)}')

    if persist:
        _persist_mappings(mappings, work_folder)
        _persist_dataset(df_c, work_folder, 'mqtt_iot_ids2020')
        _persist_subsets(X_train, X_test, y_train, y_test, work_folder)

    if return_X_y:
        return X_train, X_test, y_train, y_test
    else:
        return df_c


if __name__ == "__main__":
    load_mqtt_iot_ids2020(None, True, False)
