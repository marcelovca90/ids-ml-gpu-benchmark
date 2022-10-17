#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import logging
import os
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)

from model_suggester import CLASSIFIER_NAMES, get_optimized_suggestion

# In[2]:


QUICK_RUN      = False
N_TRIALS       = 100
TIMEOUT        = 60*60*4
FILE_PREFIX    = 'sequential-noencoding-optimized'
DATASET_FOLDER = '../datasets/DATASET_NAME/'
n_cpus          = os.cpu_count()
n_parallel      = int(n_cpus / 2)
uuid            = str(uuid.uuid4())[:8]


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


OPTUNA_EARLY_STOPING = 20

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None


def early_stopping(study, trial):
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


# In[3]:


print(f"Sequential-NoEncoding-Optimized.py - Execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[4]:

def load_mappings(filename='mappings.json'):
    base_folder = os.path.join(os.path.dirname(__file__), DATASET_FOLDER)
    full_filename = os.path.join(base_folder, filename)
    with open(full_filename, 'r') as fp:
        return json.load(fp)

def load_csv(filename):
    base_folder = os.path.join(os.path.dirname(__file__), DATASET_FOLDER)
    full_filename = os.path.join(base_folder, filename)
    df = pd.read_csv(filepath_or_buffer=full_filename).infer_objects().to_numpy()
    return df.ravel() if df.shape[1] == 1 else df
    
def load_hdf(filename):
    base_folder = os.path.join(os.path.dirname(__file__), DATASET_FOLDER)
    full_filename = os.path.join(base_folder, filename)
    df = pd.read_hdf(path_or_buf=full_filename).infer_objects().to_numpy()
    return df.ravel() if df.shape[1] == 1 else df


# In[5]:

mappings = load_mappings()
class_names = [str(v) for v in mappings.keys()]
class_indices = [int(k) for k in mappings.values()]

X_train, X_test, y_train, y_test = load_csv('X_train.csv'), load_csv('X_test.csv'), load_csv('y_train.csv'), load_csv('y_test.csv')
# X_train, X_test, y_train, y_test = load_hdf('X_train.h5'), load_hdf('X_test.h5'), load_hdf('y_train.h5'), load_hdf('y_test.h5')

# t = MinMaxScaler()
# t.fit(X_train)
# X_train = t.transform(X_train)
# X_test = t.transform(X_test)

print('X_train',X_train.shape,'\ny_train',y_train.shape,set(y_train))
print('X_test',X_test.shape,'\ny_test',y_test.shape,set(y_test))

if QUICK_RUN:
    N_TRIALS = 4
    TIMEOUT = 60
    limit = 128*128
    X_train = X_train[:limit]
    y_train = y_train[:limit]
    X_test = X_test[:limit]
    y_test = y_test[:limit]


# In[8]:


print(f"Optimization batch started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[ ]:


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(f"{FILE_PREFIX}_{uuid}.log", mode="w"))
optuna.logging.enable_propagation()

best_results = {}

print(f'Optimizing {len(CLASSIFIER_NAMES)} classifiers.')

for classifier_name in CLASSIFIER_NAMES:

    def objective(trial):
        
        classifier_obj = get_optimized_suggestion(X_train, y_train, classifier_name, trial)

        # print(f'Suggested model: {type(classifier_obj)} with {vars(classifier_obj)}.')
    
        classifier_obj.fit(X_train, y_train)

        y_pred = classifier_obj.predict(X_test)
        
        _accuracy                         = calculate_score('accuracy',          class_names, y_test, y_pred)
        _balanced_accuracy                = calculate_score('balanced_accuracy', class_names, y_test, y_pred)
        _f1_score_micro                   = calculate_score('f1_score_micro',    class_indices, y_test, y_pred)
        _f1_score_macro                   = calculate_score('f1_score_macro',    class_indices, y_test, y_pred)
        _f1_score_weighted                = calculate_score('f1_score_weighted', class_indices, y_test, y_pred)
        _classification_report            = {str(k): v for k,v in classification_report(y_test, y_pred, output_dict=True, zero_division=0).items()}
        _classification_report_imbalanced = {str(k): v for k,v in classification_report_imbalanced(y_test, y_pred, output_dict=True, zero_division=0).items()}

        trial.set_user_attr('accuracy',                         _accuracy)
        trial.set_user_attr('balanced_accuracy',                _balanced_accuracy)
        trial.set_user_attr('f1_score_micro',                   _f1_score_micro)
        trial.set_user_attr('f1_score_macro',                   _f1_score_macro)
        trial.set_user_attr('f1_score_weighted',                _f1_score_weighted)
        trial.set_user_attr('classification_report',            json.dumps(_classification_report, default=str))
        trial.set_user_attr('classification_report_imbalanced', json.dumps(_classification_report_imbalanced, default=str))

        try:
            return _f1_score_weighted
        except Exception as e:
            print(f'Unable to return score for {classifier_name}. Reason: {e}')
            return 0.0
    
    study = optuna.create_study(study_name=f"{classifier_name}_{uuid}", storage="mysql://root:root@localhost/optuna", load_if_exists=True, direction='maximize')

    print(f'--------------------------------------------------------------------------------')
    print(f'-------------------- {classifier_name.center(38)} --------------------')
    print(f'--------------------------------------------------------------------------------')
    
    print(f"Study optimization started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    
    try:
        if classifier_name in ['LinearSVC', 'LinearSVC_AdditiveChi2Sampler', 'LinearSVC_Nystroem', 'LinearSVC_PolynomialCountSketch', 'LinearSVC_RBFSampler', 'TabNetClassifier']:
            n_jobs = 1    
        elif classifier_name in ['CatBoostClassifier']:
            n_jobs = 2
        elif classifier_name in ['XGBClassifier']:
            n_jobs = 4
        else:
            n_jobs = n_parallel

        study.optimize(objective, timeout=TIMEOUT, n_trials=N_TRIALS, n_jobs=n_parallel, callbacks=[early_stopping], catch=(ValueError,), gc_after_trial=True)        

    except EarlyStoppingExceeded:
        print(f'EarlyStopping exceeded for {classifier_name}; no new best scores on iters {OPTUNA_EARLY_STOPING}.')
    except Exception as e:
        print(f"Study with classifier {classifier_name} failed. Reason: {e}")
    print(f"Study optimization finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    
    # optuna.visualization.plot_optimization_history(study).show()
    
    if classifier_name not in best_results.keys():
        best_results[classifier_name] = []
    
    try:
        best_results[classifier_name] = \
            [{'number': trial.number, 'values': trial.values, 'params': trial.params, 'user_attrs': trial.user_attrs} for trial in study.get_trials()]
        with open(f'{FILE_PREFIX}_{uuid}_{classifier_name}.json', 'w') as fp:
            json.dump(best_results[classifier_name], fp)
    except Exception as e:
        best_results[classifier_name] = \
            [{'number': None,         'values': None,         'params': None,         'user_attrs': None}             for trial in study.get_trials()]
        print(f'Unable to persist results for {classifier_name}. Reason: {e}')

# In[ ]:


print(f"Optimization batch finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# In[ ]:


# for each classifier, sort results according to score (descending) and remove results without report
# best_results = dict(sorted(best_results.items()))
for key,value in best_results.items():
    #_classification_report = json.loads(value['_classification_report'])
    #_classification_report_imbalanced = json.loads(value['_classification_report_imbalanced'])
    try:
        best_results[key] = sorted([v for v in best_results[key]], key=lambda d: d['user_attrs']['f1_score_weighted'], reverse=True)
    except Exception as e:
        print(f'Unable to set best_results[{key}]. Reason: {e}')
        best_results[key] = 0.0

# persist results to filesystem
with open(f'{FILE_PREFIX}_{uuid}.json', 'w') as fp:
    json.dump(best_results, fp)


# In[ ]:


# plot the best result for each classifier
plt.figure(figsize=(32,24))
plt.rcParams['font.size'] = '16'
idx = 0
for key, value in best_results.items():
    name_i = key
    value_i = 0.0
    try:
        value_i = value[0]['user_attrs']['f1_score_weighted']
    except Exception as e:
        print(f'Unable to retrieve best result for {key}. Reason: {e}')
        value_i = 0.0
    plt.bar(name_i,value_i)
    plt.text(idx-0.35,value_i+0.01,f'{100*value_i:.2f}%')
    idx += 1

# plot the best result for every classifier and the mean line
try:
    mean = np.mean([v[0]['user_attrs']['f1_score_weighted'] for v in best_results.values()])
except Exception as e:
    print(f'Unable to calculate the mean score. Reason: {e}')
    mean = 0.0
plt.axhline(mean, color='black', linestyle='--')
plt.xticks(rotation=30, ha='right')
plt.xticks(range(0,len(best_results)),best_results.keys())
plt.yticks(rotation=30, ha='right')
plt.yticks(np.linspace(0,1,11))
plt.ylim(0,1.05)
plt.savefig(f'{FILE_PREFIX}_{uuid}.png')
plt.show(block=False)


# In[ ]:


print(f"Sequential-NoEncoding-Optimized.py - Execution finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

