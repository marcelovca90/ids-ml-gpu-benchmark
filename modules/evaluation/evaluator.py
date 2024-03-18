import os

from pycaret.classification import *
from pycaret.containers.models.classification import get_all_model_containers
from pycaret.utils.generic import get_model_id
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from typing_extensions import Self

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline


def _f1_score_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def _f1_score_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def _f1_score_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def _build_model_params(model_id, use_class_weights=False, custom_engine=None):
    model_params = {}
    if custom_engine != 'default':
        model_params['engine'] = custom_engine
    if model_id == 'catboost':
        model_params['auto_class_weights'] = \
            'Balanced' if use_class_weights else 'None'
    else:
        model_params['class_weight'] = \
            'balanced' if use_class_weights else None
    return model_params


def _build_tune_grid(exp, model, use_class_weights):
    model_id = get_model_id(model, get_all_model_containers(exp))
    if model_id:
        tune_grid = get_all_model_containers(exp)[model_id].tune_grid
        if model_id == 'catboost':
            del tune_grid['eta']
            tune_grid['learning_rate'] = [0.02, 0.03, 0.04]
            tune_grid['auto_class_weights'] = \
                ['Balanced'] if use_class_weights else ['None']
    else:
        tune_grid = \
            {'class_weight': ['balanced'] if use_class_weights else [None]}
    return tune_grid


@function_call_logger
def run(
        bpp: BasePreprocessingPipeline,
        fix_imbalance: bool,
        use_class_weights: bool) -> Self:

    lb_filename_preffix = os.path.join(
        os.getcwd(),
        f'{bpp.__class__.__name__}_FI={fix_imbalance}_CW={use_class_weights}')
    sort_col, drop_subset = 'F1 (weighted)', 'Model Name'

    exp = ClassificationExperiment()
    exp.setup(
        data=bpp.data,
        target=bpp.target,
        train_size=0.8,
        fold=5,
        fold_strategy='stratifiedkfold',
        fold_shuffle=True,
        feature_selection=True,
        n_features_to_select=0.8,
        use_gpu=False,
        session_id=bpp.seed
    )

    exp.add_metric('f1_score_micro', 'F1 (micro)', _f1_score_micro)
    exp.add_metric('f1_score_macro', 'F1 (macro)', _f1_score_macro)
    exp.add_metric('f1_score_weighted', 'F1 (weighted)', _f1_score_weighted)

    base_models_ids = \
        ['dt', 'et', 'lightgbm', 'lr', 'rf', 'ridge', 'svm', 'xgboost']

    base_models = []
    for model_id in base_models_ids:
        log_print(f'Creating model {model_id.upper()}...')
        try:
            model = exp.create_model(
                model_id, **_build_model_params(
                    model_id,
                    use_class_weights=use_class_weights,
                    custom_engine='sklearnex'))
        except Exception as e:
            model = exp.create_model(
                model_id, **_build_model_params(
                    model_id,
                    use_class_weights=use_class_weights,
                    custom_engine=None))
        base_models.append(model)

    # log_print(f'Creating model HGBC...')
    # base_models.append(exp.create_model(
    #     HistGradientBoostingClassifier(), class_weight=None))

    # log_print(f'Creating model PAC...')
    # base_models.append(exp.create_model(
    #     PassiveAggressiveClassifier(), class_weight=None))

    base_lb = exp.get_leaderboard(model_only=True).sort_values(
        by=[sort_col], ascending=False).drop_duplicates(subset=[drop_subset])
    base_lb.to_excel(lb_filename_preffix + '_base.xlsx')

    tuned_models = []
    for model in base_models:
        print(f'\nTuning model {model.__class__.__name__}...')
        tuned_model = exp.tune_model(
            model,
            search_library='optuna',
            search_algorithm='tpe',
            custom_grid=_build_tune_grid(exp, model, use_class_weights))
        tuned_models.append(tuned_model)

    tuned_lb = exp.get_leaderboard(model_only=True).sort_values(
        by=[sort_col], ascending=False).drop_duplicates(subset=[drop_subset])
    tuned_lb.to_excel(lb_filename_preffix + '_hpo.xlsx')
