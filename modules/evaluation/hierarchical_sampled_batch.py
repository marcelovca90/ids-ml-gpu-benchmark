import inspect

import numpy.ma as ma
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import all_estimators
from sklearn.utils.class_weight import compute_sample_weight
from sklearn_hierarchical_classification.classifier import \
    HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import (h_fbeta_score,
                                                         multi_labeled)
from tqdm import tqdm as tqdm_wrapper

from modules.logging.logger import log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline


class CustomHierarchicalClassifier:

    @staticmethod
    def run(dp: BasePreprocessingPipeline, subsample_size=None):

        # Get all scikit-learn classifiers
        all_classifiers = all_estimators(type_filter='classifier')

        # List to store classifiers with predict_proba
        classifiers_with_predict_proba = []

        # Check if each classifier has predict_proba method
        for name, ClassifierClass in all_classifiers:
            constructor_params = inspect.signature(ClassifierClass).parameters
            invalid_params = ['base_estimator', 'estimator', 'estimators']
            if not set(constructor_params).intersection(invalid_params) and \
                    hasattr(ClassifierClass(), 'predict_proba'):
                classifiers_with_predict_proba.append((name, ClassifierClass))

        # Instantiate classifiers with predict_proba and set random_state if available
        for name, ClassifierClass in classifiers_with_predict_proba:
            # Get the parameters of the classifier's constructor
            constructor_params = inspect.signature(ClassifierClass).parameters
            # Check if random_state exists as a parameter in the constructor
            if 'random_state' in constructor_params:
                # Create an instance of the classifier with random_state set
                base_estimator = ClassifierClass(random_state=42)
            else:
                # Create an instance of the classifier without setting random_state
                base_estimator = ClassifierClass()

            # Use the base_estimator for further use or training.
            log_print(f"Instantiated {name} classifier => {base_estimator}.")

            try:

                # "0": "Attack",
                # "2": "C&C",
                # "3": "C&C-FileDownload",
                # "4": "C&C-HeartBeat",
                # "5": "C&C-HeartBeat-Attack",
                # "6": "C&C-HeartBeat-FileDownload",
                # "7": "C&C-Mirai",
                # "8": "C&C-PartOfAHorizontalPortScan",
                # "9": "C&C-Torii",
                # "10": "DDoS",
                # "11": "FileDownload",
                # "12": "Okiru",
                # "13": "Okiru-Attack",
                # "14": "PartOfAHorizontalPortScan",
                # "15": "PartOfAHorizontalPortScan-Attack"

                if subsample_size:

                    # Get common indices between the two DataFrames
                    common_indices = dp.X_train.index.intersection(
                        dp.y_train.index)
                    # Sample the common indices from both DataFrames
                    sample_size = int(0.8 * subsample_size)
                    dp.X_train = dp.X_train.loc[common_indices].sample(
                        n=sample_size)
                    dp.y_train = dp.y_train.loc[common_indices].sample(
                        n=sample_size)

                    # Get common indices between the two DataFrames
                    common_indices = dp.X_test.index.intersection(
                        dp.y_test.index)
                    # Sample the common indices from both DataFrames
                    sample_size = int(0.2 * subsample_size)
                    dp.X_test = dp.X_test.loc[common_indices].sample(
                        n=sample_size)
                    dp.y_test = dp.y_test.loc[common_indices].sample(
                        n=sample_size)

                # change dtypes
                dp.y_train = dp.y_train.astype(str)
                dp.y_test = dp.y_test.astype(str)

                class_hierarchy = {
                    ROOT: ["1", "A"],
                    "A": ["0", "10", "11", "B"],
                    "B": ["C", "D", "E"],
                    "C": ["2", "3", "F", "7", "8", "9"],
                    "D": ["12", "13"],
                    "E": ["14", "15"],
                    "F": ["4", "5", "6"]
                }

                clf = HierarchicalClassifier(
                    base_estimator=base_estimator,
                    class_hierarchy=class_hierarchy,
                    progress_wrapper=tqdm_wrapper
                )

                sample_weights = compute_sample_weight(
                    class_weight='balanced', y=dp.y_train)
                log_print(f"Sample Weights: {sample_weights}")

                clf.fit(dp.X_train, dp.y_train)

                y_pred = clf.predict(dp.X_test)

                log_print(classification_report(
                    dp.y_test.astype(str), y_pred))

            except Exception as e:
                log_print(f'{base_estimator} failed. Reason: {e}')
