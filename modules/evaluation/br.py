import networkx as nx
from networkx import stochastic_block_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skmultilearn.cluster import (IGraphLabelGraphClusterer,
                                  LabelCooccurrenceGraphBuilder,
                                  NetworkXLabelGraphClusterer)
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
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

        # # Get common indices between two test DataFrames and sample them
        common_idx = dp.X_test.index.intersection(dp.y_test.index)
        test_sample_size = 20_000
        dp.X_test = dp.X_test.loc[common_idx].sample(n=test_sample_size)
        dp.y_test = dp.y_test.loc[common_idx].sample(n=test_sample_size)

        # Convert DataFrames to dense numpy arrays
        X_train_dense = dp.X_train.to_numpy()
        y_train_dense = dp.y_train.to_numpy()
        X_test_dense = dp.X_test.to_numpy()
        y_test_dense = dp.y_test.to_numpy()

        # construct base forest classifier
        base_classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

        # construct a graph builder that will include
        # label relations weighted by how many times they
        # co-occurred in the data, without self-edges
        graph_builder = LabelCooccurrenceGraphBuilder(
            weighted=True,
            include_self_edges=False
        )

        # setup problem transformation approach with sparse matrices for random forest
        problem_transform_classifier = LabelPowerset(classifier=base_classifier,
                                                     require_dense=[False, False])

        # setup the clusterer to use, we selected the fast greedy modularity-maximization approach
        # clusterer = IGraphLabelGraphClusterer(
        #    graph_builder=graph_builder, method='fastgreedy')
        clusterer = NetworkXLabelGraphClusterer(
            graph_builder=graph_builder, method='louvain')

        # setup the ensemble metaclassifier
        classifier = LabelSpacePartitioningClassifier(
            problem_transform_classifier, clusterer)

        # train
        classifier.fit(X_train_dense, y_train_dense)

        # predict
        y_pred = classifier.predict(X_test_dense).toarray()

        print(classification_report(y_test_dense, y_pred))
