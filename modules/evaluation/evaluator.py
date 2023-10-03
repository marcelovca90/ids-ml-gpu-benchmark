import os

from pycaret.classification import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from typing_extensions import Self

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline


def f1_score_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def f1_score_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def f1_score_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def build_model_params(model_id, use_class_weights=False, custom_engine=None):
    model_params = {}
    if custom_engine != 'default':
        model_params['engine'] = custom_engine
    if model_id == 'catboost':
        model_params['auto_class_weights'] = 'Balanced' if use_class_weights else 'None'
    else:
        model_params['class_weight'] = 'balanced' if use_class_weights else None
    return model_params


@function_call_logger
def baseline(dp: BasePreprocessingPipeline, use_class_weights: bool) -> Self:

    exp = ClassificationExperiment()
    exp.setup(
        data=dp.data,
        target=dp.target,
        train_size=0.8,
        fold=5,
        fold_strategy='stratifiedkfold',
        fold_shuffle=True,
        low_variance_threshold=0.0,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.9,
        use_gpu=False,
        session_id=dp.seed
    )

    exp.add_metric('f1_score_micro', 'F1 (micro)', f1_score_micro)
    exp.add_metric('f1_score_macro', 'F1 (macro)', f1_score_macro)
    exp.add_metric('f1_score_weighted', 'F1 (weighted)', f1_score_weighted)

    base_models_ids = ['dt', 'et', 'lightgbm',
                       'lr', 'rf', 'ridge', 'svm', 'xgboost']

    base_models = []

    for model_id in base_models_ids:
        log_print(f'Creating model {model_id.upper()}...')
        try:
            model = exp.create_model(
                model_id, **build_model_params(model_id, use_class_weights=use_class_weights, custom_engine='sklearnex'))
        except Exception as e:
            model = exp.create_model(
                model_id, **build_model_params(model_id, use_class_weights=use_class_weights, custom_engine=None))
        base_models.append(model)

    log_print(f'Creating model HGBC...')
    base_models.append(exp.create_model(
        HistGradientBoostingClassifier(), class_weight=None))

    log_print(f'Creating model PAC...')
    base_models.append(exp.create_model(
        PassiveAggressiveClassifier(), class_weight=None))

    base_lb = exp.get_leaderboard(model_only=True).sort_values(
        by=['F1 (weighted)'], ascending=False).drop_duplicates(subset=['Model Name'])

    filename = f'{dp.__class__.__name__}_{use_class_weights}_base.xlsx'
    base_lb.to_excel(os.path.join(os.getcwd(), filename))
