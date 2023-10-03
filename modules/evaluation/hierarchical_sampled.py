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

from modules.preprocessing.preprocessor import BasePreprocessingPipeline


class CustomHierarchicalClassifier:

    @staticmethod
    def get_estimators():

        # Get all scikit-learn classifiers
        all_classifiers = all_estimators(type_filter='classifier')

        # List to store classifiers with predict_proba
        classifiers_with_predict_proba = []

        # Check if each classifier has predict_proba method
        for name, ClassifierClass in all_classifiers:
            if hasattr(ClassifierClass(), 'predict_proba'):
                classifiers_with_predict_proba.append((name, ClassifierClass))
        print(f"List of classifiers: {classifiers_with_predict_proba}")

        # Instantiate classifiers with predict_proba
        for name, ClassifierClass in classifiers_with_predict_proba:
            classifier_instance = ClassifierClass()
            # Use the classifier_instance for further use or training.
            print(f"Instantiated {name} classifier => {classifier_instance}.")

    @staticmethod
    def run(dp: BasePreprocessingPipeline):

        base_estimator = DecisionTreeClassifier(random_state=42)

        class_hierarchy = {
            ROOT: ["A", "0"],
            "A": ["1", "2"]
        }

        clf = HierarchicalClassifier(
            base_estimator=base_estimator,
            class_hierarchy=class_hierarchy,
            progress_wrapper=tqdm_wrapper
        )

        # Get common indices between the two DataFrames
        common_indices = dp.X_train.index.intersection(dp.y_train.index)
        # Sample the common indices from both DataFrames
        sample_size = 80_000
        X_train_sampled = dp.X_train.loc[common_indices].sample(n=sample_size)
        y_train_sampled = dp.y_train.loc[common_indices].sample(
            n=sample_size).astype(str)

        sample_weights = compute_sample_weight(
            class_weight='balanced', y=y_train_sampled)
        print(f"Sample Weights: {sample_weights}")

        clf.fit(X_train_sampled, y_train_sampled)

        # Get common indices between the two DataFrames
        common_indices = dp.X_test.index.intersection(dp.y_test.index)
        # Sample the common indices from both DataFrames
        sample_size = 20_000
        X_test_sampled = dp.X_test.loc[common_indices].sample(n=sample_size)
        y_test_sampled = dp.y_test.loc[common_indices].sample(
            n=sample_size).astype(str)

        y_pred = clf.predict(X_test_sampled)

        print(classification_report(y_test_sampled.astype(str), y_pred))
