import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import ClassifierChain

from modules.preprocessing.preprocessor import BasePreprocessingPipeline


class HMC():

    @staticmethod
    def run(dp: BasePreprocessingPipeline):

        # Define the class hierarchy matrix
        class_hierarchy = np.array([
            [1, 0, 0],  # A
            [1, 0, 0],  # B
            [0, 1, 0],  # C
        ])

        # Get common indices between two train DataFrames and sample them
        common_idx = dp.X_train.index.intersection(dp.y_train.index)
        train_sample_size = 80_000
        dp.X_train = dp.X_train.loc[common_idx].sample(
            n=train_sample_size).to_numpy()
        dp.y_train = dp.y_train.loc[common_idx].sample(
            n=train_sample_size).to_numpy()

        # Get common indices between two test DataFrames and sample them
        common_idx = dp.X_test.index.intersection(dp.y_test.index)
        test_sample_size = 20_000
        dp.X_test = dp.X_test.loc[common_idx].sample(
            n=test_sample_size).to_numpy()
        dp.y_test = dp.y_test.loc[common_idx].sample(
            n=test_sample_size).to_numpy()

        # Build the hierarchical classifier
        # Create the top-level classifier using RandomForestClassifier
        top_level_classifier = RandomForestClassifier()

        # Train the top-level classifier to distinguish A and "others"
        top_level_labels = class_hierarchy[:, 0]
        top_level_idx = np.where(dp.y_train[:, 0] == 1)[0]
        # Use NumPy array indexing
        X_train_top_level = dp.X_train[top_level_idx]
        y_train_top_level = dp.y_train[top_level_idx, 0]
        top_level_classifier.fit(X_train_top_level, y_train_top_level)

        # Make predictions at the top level
        predictions_top_level = top_level_classifier.predict(dp.X_train)

        # Train sub-level classifiers for each sub-class within "others"
        sub_level_classifiers = {}
        for i in range(class_hierarchy.shape[1] - 1):
            sub_class_indices = np.where(class_hierarchy[:, i] == 1)[0]
            sub_level_idx = np.where(dp.y_train[:, i] == 1)[0]

            # Make sure we have more than one sub_class
            if len(sub_class_indices) > 1:
                # Use NumPy array indexing
                X_train_sub_level = dp.X_train[sub_level_idx]
                # Use [:, None]
                y_train_sub_level = dp.y_train[sub_level_idx][:,
                                                              sub_class_indices][:, None]
                sub_class_classifier.fit(X_train_sub_level, y_train_sub_level)
                sub_level_classifiers[i] = sub_class_classifier

        # Make predictions at the sub-levels
        predictions_sub_levels = []
        for i in range(class_hierarchy.shape[1] - 1):
            sub_class_idx = np.where(class_hierarchy[:, i] == 1)[0]
            sub_class_classifier = sub_level_classifiers[i]
            sub_level_idx = np.where(dp.y_train[:, i] == 1)[0]
            X_train_sub_level = dp.X_train[sub_level_idx]
            sub_level_predictions = sub_class_classifier.predict(
                X_train_sub_level)
            predictions_sub_levels.append(sub_level_predictions)

        # Combine predictions from all levels
        final_predictions = np.zeros_like(dp.y_train)
        final_predictions[:, 0] = predictions_top_level[:, 0]
        for i in range(class_hierarchy.shape[1] - 1):
            sub_class_idx = np.where(class_hierarchy[:, i] == 1)[0]
            sub_level_idx = np.where(dp.y_train[:, i] == 1)[0]
            final_predictions[sub_level_idx,
                              sub_class_idx] = predictions_sub_levels[i]

        print(final_predictions)
