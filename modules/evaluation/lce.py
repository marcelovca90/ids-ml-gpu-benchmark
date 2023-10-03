import numpy as np
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.under_sampling import (InstanceHardnessThreshold,
                                     RandomUnderSampler)
from lce import LCEClassifier
from sklearn.metrics import classification_report

from modules.preprocessing.preprocessor import BasePreprocessingPipeline


class LCE():

    @staticmethod
    def run(dp: BasePreprocessingPipeline):

        # Get common indices between two train DataFrames and sample them
        common_idx = dp.X_train.index.intersection(dp.y_train.index)
        train_sample_size = 80_000
        dp.X_train = dp.X_train.loc[common_idx].sample(n=train_sample_size)
        dp.y_train = dp.y_train.loc[common_idx].sample(n=train_sample_size)

        # # Get common indices between two test DataFrames and sample them
        common_idx = dp.X_test.index.intersection(dp.y_test.index)
        test_sample_size = 20_000
        dp.X_test = dp.X_test.loc[common_idx].sample(n=test_sample_size)
        dp.y_test = dp.y_test.loc[common_idx].sample(n=test_sample_size)

        # Convert to numpy dtypes
        dp.X_train = dp.X_train.to_numpy()
        dp.y_train = dp.y_train.to_numpy()
        dp.X_test = dp.X_test.to_numpy()
        dp.y_test = dp.y_test.to_numpy()

        unique_labels, label_counts = np.unique(dp.y_train, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            print(f"[Step 1] Label {label}: Count {count}")

        iht = InstanceHardnessThreshold(
            sampling_strategy='all', random_state=42, n_jobs=-1)
        dp.X_train, dp.y_train = iht.fit_resample(dp.X_train, dp.y_train)
        unique_labels, label_counts = np.unique(dp.y_train, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            print(f"[Step 2] Label {label}: Count {count}")

        rus = RandomUnderSampler(random_state=42)
        dp.X_train, dp.y_train = rus.fit_resample(dp.X_train, dp.y_train)
        unique_labels, label_counts = np.unique(dp.y_train, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            print(f"[Step 3] Label {label}: Count {count}")

        clf = LCEClassifier(n_jobs=-1, random_state=42, verbose=100)

        clf.fit(dp.X_train, dp.y_train)

        y_pred = clf.predict(dp.X_test)

        print(classification_report(dp.y_test, y_pred))
