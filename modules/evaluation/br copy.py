# Import BinaryRelevance from skmultilearn
# Import SVC classifier from sklearn
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import LabelPowerset

from modules.preprocessing.preprocessor import BasePreprocessingPipeline


class BR():

    @staticmethod
    def run(dp: BasePreprocessingPipeline):

        # Get common indices between two train DataFrames and sample them
        common_idx = dp.X_train.index.intersection(dp.y_train.index)
        train_sample_size = 80_000
        dp.X_train = dp.X_train.loc[common_idx].sample(n=train_sample_size)
        dp.y_train = dp.y_train.loc[common_idx].sample(n=train_sample_size)

        # Get common indices between two test DataFrames and sample them
        common_idx = dp.X_test.index.intersection(dp.y_test.index)
        test_sample_size = 20_000
        dp.X_test = dp.X_test.loc[common_idx].sample(n=test_sample_size)
        dp.y_test = dp.y_test.loc[common_idx].sample(n=test_sample_size)

        # Convert DataFrames to dense numpy arrays
        X_train_dense = dp.X_train.to_numpy()
        y_train_dense = dp.y_train.to_numpy()
        X_test_dense = dp.X_test.to_numpy()
        y_test_dense = dp.y_test.to_numpy()

        # Setup the classifier using Label Powerset
        classifier = LabelPowerset(
            classifier=DecisionTreeClassifier(), require_dense=[False, True])

        # Train
        classifier.fit(X_train_dense, y_train_dense)

        # Predict
        y_pred = classifier.predict(X_test_dense).toarray()

        print(classification_report(y_test_dense, y_pred))
