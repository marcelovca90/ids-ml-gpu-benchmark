import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def calculate_score(metric_name, labels, y_test, y_pred):
    if metric_name == 'accuracy': 
        return accuracy_score(y_test, y_pred)
    elif metric_name == 'balanced_accuracy': 
        return balanced_accuracy_score(y_test, y_pred)
    elif metric_name == 'f1_score_micro': 
        return f1_score(y_test, y_pred, labels=labels, average='micro')
    elif metric_name == 'f1_score_macro': 
        return f1_score(y_test, y_pred, labels=labels, average='macro')
    elif metric_name == 'f1_score_weighted': 
        return f1_score(y_test, y_pred, labels=labels, average='weighted')
    else:
        raise ValueError('Invalid metric name.')


def load_mappings(dataset_folder, filename='mappings.json'):
    base_folder = os.path.join(os.path.dirname(__file__), dataset_folder)
    full_filename = os.path.join(base_folder, filename)
    with open(full_filename, 'r') as fp:
        mappings = json.load(fp)
        class_names = [str(v) for v in mappings.keys()]
        class_indices = [int(k) for k in mappings.values()]
        return class_names,class_indices


def load_csv(dataset_folder, filename):
    base_folder = os.path.join(os.path.dirname(__file__), dataset_folder)
    full_filename = os.path.join(base_folder, filename)
    df = pd.read_csv(filepath_or_buffer=full_filename).infer_objects().to_numpy()
    return df.ravel() if df.shape[1] == 1 else df


def load_hdf(dataset_folder, filename):
    base_folder = os.path.join(os.path.dirname(__file__), dataset_folder)
    full_filename = os.path.join(base_folder, filename)
    df = pd.read_hdf(path_or_buf=full_filename).infer_objects().to_numpy()
    return df.ravel() if df.shape[1] == 1 else df

def truncate(data, limit=128*128):
    return data[:int(limit*0.8)]

def sort_results(best_results):
    # best_results = dict(sorted(best_results.items()))
    for key,value in best_results.items():
        #_classification_report = json.loads(value['_classification_report'])
        #_classification_report_imbalanced = json.loads(value['_classification_report_imbalanced'])
        try:
            best_results[key] = sorted([v for v in best_results[key]], key=lambda d: d['user_attrs']['f1_score_weighted'], reverse=True) 
        except Exception as e:
            print(f'Unable to set best_results[{key}]. Reason: {e}')
            best_results[key] = 0.0
    return best_results

def persist_best_results(file_prefix, uuid, best_results):
    with open(f'{file_prefix}_{uuid}.json', 'w') as fp:
        json.dump(best_results, fp)

def plot_best_and_mean_results(file_prefix, uuid, best_results):
    # plot the best result for each classifier
    plt.figure(figsize=(32,24))
    plt.rcParams['font.size'] = '16'
    idx = 0
    for key, value in best_results.items():
        name_i = key
        try:
            value_i = value[0]['user_attrs']['f1_score_weighted']
        except Exception as e:
            print(f'Unable to retrieve best result for {key}. Reason: {e}')
            value_i = 0.0
        plt.bar(name_i,value_i)
        plt.text(idx-0.35,value_i+0.01,f'{100*value_i:.2f}%')
        idx += 1

    # calculate and plot the mean line
    values_for_mean = []
    for v in best_results.values():
        try:
            values_for_mean.append(v[0]['user_attrs']['f1_score_weighted'])
        except Exception as e:
            print(f'Unable to retrieve value for mean score. Reason: {e}')
            values_for_mean.append(0.0)
    mean = np.mean(values_for_mean)
    plt.axhline(mean, color='black', linestyle='--')
    plt.xticks(rotation=30, ha='right')
    plt.xticks(range(0,len(best_results)),best_results.keys())
    plt.yticks(rotation=30, ha='right')
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0,1.05)
    plt.savefig(f'{file_prefix}_{uuid}.png')
    plt.show(block=False)

EARLY_STOPPING_PATIENCE = 20

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = EARLY_STOPPING_PATIENCE
    early_stop_count = 0
    best_score = None

def early_stopping_callback(study, trial):
    if EarlyStoppingExceeded.best_score == None:
      EarlyStoppingExceeded.best_score = study.best_value

    if study.best_value < EarlyStoppingExceeded.best_score:
        EarlyStoppingExceeded.best_score = study.best_value
        EarlyStoppingExceeded.early_stop_count = 0
    else:
      if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
            EarlyStoppingExceeded.early_stop_count = 0
            EarlyStoppingExceeded.best_score = None
            raise EarlyStoppingExceeded()
      else:
            EarlyStoppingExceeded.early_stop_count=EarlyStoppingExceeded.early_stop_count+1
    # print(f'EarlyStop counter: {EarlyStoppingExceeded.early_stop_count}, Best score: {study.best_value} and {EarlyStoppingExceeded.best_score}')
    return
