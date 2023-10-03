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

        sample_weights = compute_sample_weight('balanced', dp.y_train)
        print(f"Sample Weights: {sample_weights}")

        clf.fit(dp.X_train, dp.y_train)

        y_pred = clf.predict(dp.X_test)

        print(classification_report(dp.y_test.astype(str), y_pred))

        mlb = MultiLabelBinarizer()
        dp.y_test = mlb.fit_transform(dp.y_test.to_numpy())
        y_pred = mlb.fit_transform(y_pred)

        # Demonstrate using our hierarchical metrics module with MLB wrapper
        with multi_labeled(dp.y_test, y_pred, clf.graph_) as \
                (y_test_, y_pred_, graph_):
            h_fbeta = h_fbeta_score(y_test_, y_pred_, graph_,)
            print("h_fbeta_score: ", h_fbeta)
