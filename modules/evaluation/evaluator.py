import json
import os
import uuid

import optuna
# from sklearnex import patch_sklearn, unpatch_sklearn
# patch_sklearn(global_patch=True)
# unpatch_sklearn()
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report
from sklearn.tree import ExtraTreeClassifier
from typing_extensions import Self

from modules.evaluation.utils import (calculate_score, persist_best_results,
                                      plot_best_and_mean_results, sort_results)
from modules.logging.logger import function_call_logger, log_print
from modules.modelling.suggester import (CLASSIFIER_NAMES,
                                         get_baseline_suggestion)
from modules.preprocessing.preprocessor import BasePreprocessingPipeline


@function_call_logger
def warmup(dp: BasePreprocessingPipeline) -> Self:
    log_print(f"Performing warmup evaluation...")
    cls = ExtraTreeClassifier(random_state=dp.seed)
    cls.fit(dp.X_train, dp.y_train)
    log_print(f"Training Score: {cls.score(dp.X_train, dp.y_train)}")
    log_print(f"Test Score    : {cls.score(dp.X_test, dp.y_test)}")


@function_call_logger
def baseline(dp: BasePreprocessingPipeline) -> Self:

    mode = 'baseline'
    uuid_str = str(uuid.uuid4())[:8]
    file_prefix = os.path.join(
        os.getcwd(), 'results', dp.name, mode, uuid_str)
    os.makedirs(file_prefix, exist_ok=True)

    X_train, y_train = dp.X_train.to_numpy(), dp.y_train.to_numpy()
    class_names = [str(v) for v in dp.metadata['target_mappings'].keys()]
    class_indices = [int(k) for k in dp.metadata['target_mappings'].values()]
    best_results = {}

    for clf_name in CLASSIFIER_NAMES:

        def objective(trial):

            classifier_obj = get_baseline_suggestion(
                X_train, y_train, clf_name, trial)

            log_print(f'Suggested model: {type(classifier_obj)} ' +
                      f'with {vars(classifier_obj)}.')

            classifier_obj.fit(X_train, y_train)

            y_pred = classifier_obj.predict(dp.X_test)

            _accuracy = calculate_score(
                'accuracy', class_names, dp.y_test, y_pred)
            _balanced_accuracy = calculate_score(
                'balanced_accuracy', class_names, dp.y_test, y_pred)
            _f1_score_micro = calculate_score(
                'f1_score_micro', class_indices, dp.y_test, y_pred)
            _f1_score_macro = calculate_score(
                'f1_score_macro', class_indices, dp.y_test, y_pred)
            _f1_score_weighted = calculate_score(
                'f1_score_weighted', class_indices, dp.y_test, y_pred)
            _clf_report = \
                {str(k): v for k, v in classification_report(
                    dp.y_test, y_pred, output_dict=True, zero_division=0)
                    .items()}
            _clf_report_imb = \
                {str(k): v for k, v in classification_report_imbalanced(
                    dp.y_test, y_pred, output_dict=True, zero_division=0)
                    .items()}

            trial.set_user_attr('accuracy', _accuracy)
            trial.set_user_attr('balanced_accuracy', _balanced_accuracy)
            trial.set_user_attr('f1_score_micro', _f1_score_micro)
            trial.set_user_attr('f1_score_macro', _f1_score_macro)
            trial.set_user_attr('f1_score_weighted', _f1_score_weighted)
            trial.set_user_attr('classification_report',
                                json.dumps(_clf_report, default=str))
            trial.set_user_attr('classification_report_imbalanced',
                                json.dumps(_clf_report_imb, default=str))

            try:
                return _f1_score_weighted
            except Exception as e:
                log_print(
                    f'Unable to return score for {clf_name}. Reason: {e}')
                return 0.0

        log_print(f'--------------------------------------------------')
        log_print(f'-------- {clf_name.center(32)} --------')
        log_print(f'--------------------------------------------------')

        study = optuna.create_study(
            study_name=f"{uuid_str}_{clf_name}",
            storage="mysql://root:root@localhost/optuna",
            load_if_exists=True,
            direction='maximize')

        log_print(f"Study started.")
        try:
            study.optimize(objective, timeout=None, n_trials=1, n_jobs=1,
                           catch=(ValueError,), gc_after_trial=True)
        except Exception as e:
            log_print(
                f"Study with classifier {clf_name} failed. Reason: {e}")
        log_print(f"Study finished.")

        # optuna.visualization.plot_optimization_history(study).show()

        if clf_name not in best_results.keys():
            best_results[clf_name] = []

        try:
            best_results[clf_name] = [{'number': trial.number,
                                       'values': trial.values,
                                       'params': trial.params,
                                       'user_attrs': trial.user_attrs}
                                      for trial in study.get_trials()]
            json_filename = os.path.join(file_prefix, f'{clf_name}.json')
            with open(json_filename, 'w') as fp:
                json.dump(best_results[clf_name], fp)

        except Exception as e:
            best_results[clf_name] = \
                [{'number': None, 'values': None, 'params': None,
                    'user_attrs': None} for trial in study.get_trials()]
            log_print(
                f'Unable to persist results for {clf_name}. Reason: {e}')

    # for each classifier, sort results according to score (descending)
    # and remove results without report
    best_results = sort_results(best_results)

    # persist results to filesystem
    persist_best_results(file_prefix, uuid_str, best_results)

    # plot and persist best and mean results to filesystem
    plot_best_and_mean_results(file_prefix, uuid_str, best_results)

    log_print(f"Baselining finished.")
