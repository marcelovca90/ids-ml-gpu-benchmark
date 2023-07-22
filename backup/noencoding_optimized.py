import json
import logging
import sys
import uuid

import colorlog
import optuna
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report

from model_suggester import (CLASSIFIER_NAMES, get_n_jobs,
                             get_optimized_suggestion)
from study_utils import (EARLY_STOPPING_PATIENCE, EarlyStoppingExceeded,
                         calculate_score, early_stopping_callback, load_csv,
                         load_mappings, persist_best_results,
                         plot_best_and_mean_results, sort_results, truncate)

QUICK_RUN      = False
N_TRIALS       = 100
TIMEOUT        = 60*60*4
FILE_PREFIX    = 'noencoding-optimized'
DATASET_FOLDER = '../datasets/iot_network_intrusion/macro/'
UUID           = str(uuid.uuid4())[:8]

formatter = colorlog.ColoredFormatter("%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s")
handler_stdout = logging.StreamHandler(stream=sys.stdout)
handler_stdout.setFormatter(formatter)
handler_file = logging.FileHandler(f"{FILE_PREFIX}_{UUID}.log", mode="w")
handler_file.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler_stdout)
logger.addHandler(handler_file)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

logger.info(f"Optimization for {len(CLASSIFIER_NAMES)} classifiers started.")

class_names,class_indices = load_mappings(DATASET_FOLDER)

X_train, X_test = load_csv(DATASET_FOLDER, 'X_train.csv'), load_csv(DATASET_FOLDER, 'X_test.csv')
y_train, y_test = load_csv(DATASET_FOLDER, 'y_train.csv'), load_csv(DATASET_FOLDER, 'y_test.csv')

if QUICK_RUN:
    N_TRIALS = 4
    TIMEOUT = 60
    X_train, X_test, y_train, y_test = truncate(X_train), truncate(X_test), truncate(y_train), truncate(y_test)

logger.info(f'X_train shape: {X_train.shape}; y_train shape: {y_train.shape}; y_train unique values: {set(y_train)}')
logger.info(f'X_test shape: {X_test.shape}; y_test shape: {y_test.shape}; y_test unique values: {set(y_test)}')

best_results = {}

for classifier_name in CLASSIFIER_NAMES:

    def objective(trial):
        
        classifier_obj = get_optimized_suggestion(X_train, y_train, classifier_name, trial)

        # logger.info(f'Suggested model: {type(classifier_obj)} with {vars(classifier_obj)}.')
    
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
            logger.info(f'Unable to return score for {classifier_name}. Reason: {e}')
            return 0.0

    logger.info(f'--------------------------------------------------------------------------------')
    logger.info(f'-------------------- {classifier_name.center(38)} --------------------')
    logger.info(f'--------------------------------------------------------------------------------')
    
    study = optuna.create_study(study_name=f"{classifier_name}_{UUID}", storage="mysql://root:root@localhost/optuna", load_if_exists=True, direction='maximize')

    logger.info(f"Study started.")
    try:
        n_jobs = get_n_jobs(classifier_name)
        study.optimize(objective, timeout=TIMEOUT, n_trials=N_TRIALS, n_jobs=n_jobs, catch=(ValueError,), gc_after_trial=True)
    except EarlyStoppingExceeded:
        logger.info(f'EarlyStopping exceeded for {classifier_name}; no new best scores on iters {EARLY_STOPPING_PATIENCE}.')
    except Exception as e:
        logger.info(f"Study with classifier {classifier_name} failed. Reason: {e}")
    logger.info(f"Study finished.")
    
    # optuna.visualization.plot_optimizationd_history(study).show()
    
    if classifier_name not in best_results.keys():
        best_results[classifier_name] = []
    
    try:
        best_results[classifier_name] = \
            [{'number': trial.number, 'values': trial.values, 'params': trial.params, 'user_attrs': trial.user_attrs} for trial in study.get_trials()]
        with open(f'{FILE_PREFIX}_{UUID}_{classifier_name}.json', 'w') as fp:
            json.dump(best_results[classifier_name], fp)
    except Exception as e:
        best_results[classifier_name] = \
            [{'number': None,         'values': None,         'params': None,         'user_attrs': None} for trial in study.get_trials()]
        logger.info(f'Unable to persist results for {classifier_name}. Reason: {e}')

# for each classifier, sort results according to score (descending) and remove results without report
best_results = sort_results(best_results)

# persist results to filesystem
persist_best_results(FILE_PREFIX, UUID, best_results)

# plot and persist best and mean results to filesystem
plot_best_and_mean_results(FILE_PREFIX, UUID, best_results)

logger.info(f"Optimization finished.")
