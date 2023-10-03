import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


def hierarchical_classification(X, y):
    """
    This function performs hierarchical classification using logistic regression.

    Args:
      X: The training data.
      y: The training labels.

    Returns:
      The fitted classifier.
    """

    # Get the hierarchy of classes.
    hierarchy = np.unique(y)
    hierarchy = np.sort(hierarchy)

    # Initialize the classifiers.
    classifiers = []
    for i in range(len(hierarchy) - 1):
        clf = LogisticRegression()
        classifiers.append(clf)

    # Fit the classifiers.
    for i in range(len(hierarchy) - 1):
        current_class = hierarchy[i]
        other_classes = hierarchy[i + 1:]
        y_filtered = np.where(y == current_class, 1, 0)
        clf.fit(X, y[y_filtered])

    # Initialize the metaclassifier.
    metaclf = VotingClassifier(estimators=classifiers)

    # Fit the metaclassifier.
    metaclf.fit(X, y)

    # Return the fitted classifier.
    return metaclf


if __name__ == "__main__":
    # Generate some data.
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 3, 100)

    # Fit the classifier.
    classifiers = hierarchical_classification(X, y)

    # Make some predictions.
    new_X = np.random.rand(10, 2)
    predictions = classifiers.predict(new_X)

    # Print the predictions.
    print(predictions)
