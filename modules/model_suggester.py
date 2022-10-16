from random import random
from tkinter import N

from sklearnex import patch_sklearn

patch_sklearn(global_patch=True)
import sklearn
from catboost import CatBoostClassifier
from imblearn.ensemble import (BalancedBaggingClassifier,
                               BalancedRandomForestClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)
from lightgbm import LGBMClassifier
from lightning.classification import AdaGradClassifier, CDClassifier
from pyAgrum.skbn import BNClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from skeLCS import eLCS
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.kernel_approximation import (AdditiveChi2Sampler, Nystroem,
                                          PolynomialCountSketch, RBFSampler)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, SGDClassifier)
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, ComplementNB,
                                 GaussianNB, MultinomialNB)
from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid,
                               RadiusNeighborsClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm._classes import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearnex.svm.nusvc import NuSVC
from xgboost import XGBClassifier

SEED        = 10
N_JOBS      = 1

CLASSIFIER_NAMES = [
    'AdaBoostClassifier',
    'AdaGradClassifier',
    'BalancedBaggingClassifier',
    'BalancedRandomForestClassifier',
    'BernoulliNB',
    'BNClassifier',
    #'CatBoostClassifier',
    #'CategoricalNB',
    'CDClassifier',
    'ComplementNB',
    'DecisionTreeClassifier',
    #'EasyEnsembleClassifier',
    #'eLCS',
    'ExtraTreeClassifier',
    'ExtraTreesClassifier',
    'GaussianNB',
    #'GaussianProcessClassifier',
    'HistGradientBoostingClassifier',
    #'KNeighborsClassifier',
    'LGBMClassifier',
    'LinearDiscriminantAnalysis',
    'LinearSVC',
    'LinearSVC_AdditiveChi2Sampler',
    'LinearSVC_Nystroem',
    'LinearSVC_PolynomialCountSketch',
    'LinearSVC_RBFSampler',
    #'LogisticRegression',
    'MLPClassifier',
    'MultinomialNB',
    #'NearestCentroid',
    #'NuSVC_Linear',
    #'NuSVC_RBF',
    #'NuSVC_Sigmoid',
    'PassiveAggressiveClassifier',
    'Perceptron',
    'QuadraticDiscriminantAnalysis',
    #'RadiusNeighborsClassifier',
    'RandomForestClassifier',
    'RidgeClassifier',
    'RUSBoostClassifier',
    'SGDClassifier',
    #'TabNetClassifier',
    'XGBClassifier',
]

def get_baseline_suggestion(X_train, y_train, classifier_name, trial):

    if classifier_name      == 'AdaBoostClassifier':
        classifier_obj       = AdaBoostClassifier(random_state=SEED)
    
    elif classifier_name    == 'AdaGradClassifier':
        classifier_obj       = AdaGradClassifier(random_state=SEED)
        
    elif classifier_name    == 'BalancedBaggingClassifier':
        classifier_obj       = BalancedBaggingClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'BalancedRandomForestClassifier':
        classifier_obj       = BalancedRandomForestClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'BernoulliNB':
        classifier_obj       = BernoulliNB()    
    
    elif classifier_name   == 'BNClassifier':
        classifier_obj       = BNClassifier()
    
    elif classifier_name    == 'CatBoostClassifier':
        classifier_obj       = CatBoostClassifier(random_state=SEED, thread_count=N_JOBS)

    elif classifier_name    == 'CategoricalNB':
        classifier_obj       = CategoricalNB()

    elif classifier_name    == 'CDClassifier':
        classifier_obj       = CDClassifier(random_state=SEED)

    elif classifier_name    == 'ComplementNB':
        classifier_obj       = ComplementNB()

    elif classifier_name    == 'DecisionTreeClassifier':
        classifier_obj       = DecisionTreeClassifier(random_state=SEED)

    elif classifier_name    == 'EasyEnsembleClassifier':
        classifier_obj       = EasyEnsembleClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'eLCS':
        classifier_obj       = eLCS(random_state=SEED)

    elif classifier_name    == 'ExtraTreeClassifier':
        classifier_obj       = ExtraTreeClassifier(random_state=SEED)

    elif classifier_name    == 'ExtraTreesClassifier':
        classifier_obj       = ExtraTreesClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'GaussianNB':
        classifier_obj       = GaussianNB()

    elif classifier_name    == 'GaussianProcessClassifier':
        classifier_obj       = GaussianProcessClassifier(random_state=SEED, n_jobs=N_JOBS)
    
    elif classifier_name    == 'HistGradientBoostingClassifier':
        classifier_obj       = HistGradientBoostingClassifier(random_state=SEED)
    
    elif classifier_name    == 'KNeighborsClassifier':
        classifier_obj       = KNeighborsClassifier(n_jobs=N_JOBS)

    elif classifier_name    == 'LGBMClassifier':
        classifier_obj       = LGBMClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'LinearDiscriminantAnalysis':
        classifier_obj       = LinearDiscriminantAnalysis()
    
    elif classifier_name    == 'LinearSVC':
        classifier_obj       = LinearSVC(random_state=SEED)
    
    elif classifier_name    == 'LinearSVC_AdditiveChi2Sampler':
        classifier_obj       = make_pipeline(AdditiveChi2Sampler(), LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'LinearSVC_Nystroem':
        classifier_obj       = make_pipeline(Nystroem(random_state=SEED), LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'LinearSVC_PolynomialCountSketch':
        classifier_obj       = make_pipeline(PolynomialCountSketch(random_state=SEED), LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'LinearSVC_RBFSampler':
        classifier_obj       = make_pipeline(RBFSampler(random_state=SEED), LinearSVC(random_state=SEED))

    elif classifier_name    == 'LogisticRegression':
        classifier_obj       = LogisticRegression(n_jobs=N_JOBS)

    elif classifier_name    == 'LinearSVC':
        classifier_obj       = LinearSVC(random_state=SEED)

    elif classifier_name    == 'SGDClassifier':
        classifier_obj       = SGDClassifier(random_state=SEED)

    elif classifier_name    == 'MLPClassifier':
        classifier_obj       = MLPClassifier(random_state=SEED)

    elif classifier_name    == 'MultinomialNB':
        classifier_obj       = MultinomialNB()
        
    elif classifier_name    == 'NearestCentroid':
        classifier_obj       = NearestCentroid()

    elif classifier_name    == 'NuSVC_Linear':
        classifier_obj       = NuSVC(kernel='linear', random_state=SEED)
    
    elif classifier_name    == 'NuSVC_RBF':
        classifier_obj       = NuSVC(kernel='rbf', random_state=SEED)
        
    elif classifier_name    == 'NuSVC_Sigmoid':
        classifier_obj       = NuSVC(kernel='sigmoid', random_state=SEED)

    elif classifier_name    == 'PassiveAggressiveClassifier':
        classifier_obj       = PassiveAggressiveClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'Perceptron':
        classifier_obj       = Perceptron(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'QuadraticDiscriminantAnalysis':
        classifier_obj       = QuadraticDiscriminantAnalysis()

    elif classifier_name    == 'RadiusNeighborsClassifier':
        classifier_obj       = RadiusNeighborsClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'RandomForestClassifier':
        classifier_obj       = RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS)
                                                        
    elif classifier_name    == 'RidgeClassifier':
        classifier_obj       = RidgeClassifier(random_state=SEED)
        
    elif classifier_name    == 'RUSBoostClassifier':
        classifier_obj       = RUSBoostClassifier(random_state=SEED)
    
    elif classifier_name    == 'SGDClassifier':
        classifier_obj       = SGDClassifier(random_state=SEED, n_jobs=N_JOBS)

    elif classifier_name    == 'TabNetClassifier':
        classifier_obj       = TabNetClassifier(seed=SEED)
        
    elif classifier_name    == 'XGBClassifier':
        classifier_obj       = XGBClassifier(seed=SEED, n_jobs=N_JOBS)

    return classifier_obj


def get_optimized_suggestion(X_train, y_train, classifier_name, trial):

    if classifier_name      == 'AdaBoostClassifier':
        n_estimators         = trial.suggest_int('abc_n_estimators', 25, 100, 5)
        learning_rate        = trial.suggest_loguniform('abc_learning_rate', 1e-6, 1e0)
        classifier_obj       = AdaBoostClassifier(n_estimators=n_estimators,
                                                  learning_rate=learning_rate,
                                                  random_state=SEED)
    
    elif classifier_name    == 'AdaGradClassifier':
        alpha                = trial.suggest_loguniform('agc_alpha', 1e-3, 1e3)
        l1_ratio             = trial.suggest_discrete_uniform('agc_l1_ratio', 0.0, 1.0, 0.1)
        loss                 = trial.suggest_categorical('agc_loss', ['modified_huber', 'hinge', 'smooth_hinge', 'squared_hinge', 'perceptron', 'log', 'squared'])
        classifier_obj       = AdaGradClassifier(alpha=alpha,
                                                      l1_ratio=l1_ratio,
                                                      loss=loss,
                                                      random_state=SEED)
    
    elif classifier_name    == 'BalancedBaggingClassifier':
        n_estimators         = trial.suggest_int('bbc_n_estimators', 5, 20, 1)
        max_features         = trial.suggest_int('bbc_max_features', 1, X_train.shape[1])
        bootstrap            = trial.suggest_categorical('bbc_bootstrap', [False, True])
        sampling_strategy    = trial.suggest_categorical('bbc_sampling_strategy', ['majority', 'not minority', 'not majority', 'all'])
        classifier_obj       = BalancedBaggingClassifier(n_estimators=n_estimators,
                                                         max_features=max_features,
                                                         bootstrap=bootstrap,
                                                         sampling_strategy=sampling_strategy,
                                                         random_state=SEED,
                                                         n_jobs=N_JOBS)

    elif classifier_name    == 'BalancedRandomForestClassifier':
        n_estimators         = trial.suggest_int('brf_n_estimators', 50, 200, 10)
        criterion            = trial.suggest_categorical('brf_criterion', ['gini', 'entropy'])
        min_samples_split    = trial.suggest_int('brf_min_samples_split', 2, 50)
        min_samples_leaf     = trial.suggest_int('brf_min_samples_leaf', 1, 50)
        max_features         = trial.suggest_int('brf_max_features', 1, X_train.shape[1])
        bootstrap            = trial.suggest_categorical('brf_bootstrap', [False, True])
        classifier_obj       = BalancedRandomForestClassifier(n_estimators=n_estimators,
                                                              criterion=criterion,
                                                              min_samples_split=min_samples_split,
                                                              min_samples_leaf=min_samples_leaf,
                                                              max_features=max_features,
                                                              bootstrap=bootstrap,
                                                              random_state=SEED,
                                                              n_jobs=N_JOBS)
    elif classifier_name    == 'BernoulliNB':
        alpha                = trial.suggest_discrete_uniform('bnb_alpha', 0.0, 1.0, 0.05)
        binarize             = trial.suggest_discrete_uniform('bnb_binarize', 0.0, 1.0, 0.05)
        fit_prior            = trial.suggest_categorical('bnb_fit_prior', [False, True])
        classifier_obj       = BernoulliNB(alpha=alpha,
                                           binarize=binarize,
                                           fit_prior=fit_prior)
    
    elif classifier_name   == 'BNClassifier':
        learningMethod         = trial.suggest_categorical('bnc_learningMethod', ["Chow-Liu", "NaiveBayes", "GHC", "MIIC", "TAN", "Tabu"])
        prior                  = trial.suggest_categorical('bnc_prior', ["Smoothing", "BDeu", "Dirichlet", "NoPrior"])
        scoringType            = trial.suggest_categorical('bnc_scoringType', ["AIC", "BIC", "BD", "BDeu", "K2", "Log2"])
        discretizationStrategy = trial.suggest_categorical('bnc_discretizationStrategy', ['quantile', 'uniform', 'kmeans', 'NML', 'CAIM', 'MDLP'])
        discretizationNbBins   = trial.suggest_int('bnc_discretizationNbBins', 10, 50, 5)
        classifier_obj         = BNClassifier(learningMethod=learningMethod,
                                              prior=prior,
                                              scoringType=scoringType,
                                              discretizationStrategy=discretizationStrategy,
                                              discretizationNbBins=discretizationNbBins,
                                              usePR=True)
    
    elif classifier_name    == 'CatBoostClassifier':
        iterations           = trial.suggest_categorical('cbc_iterations', [2000])
        learning_rate        = trial.suggest_loguniform('cbc_learning_rate', 1e-3, 1e0)
        sampling_frequency   = trial.suggest_categorical('cbc_sampling_frequency', ['PerTree', 'PerTreeLevel'])
        depth                = trial.suggest_int('cbc_depth', 2, 8, 2)
        verbose              = trial.suggest_categorical('cbc_verbose', [False])
        classifier_obj       = CatBoostClassifier(iterations=iterations,
                                                  learning_rate=learning_rate,
                                                  sampling_frequency=sampling_frequency,
                                                  depth=depth,
                                                  random_state=SEED,
                                                  thread_count=N_JOBS,
                                                  verbose=verbose)

    elif classifier_name    == 'CategoricalNB':
        alpha                = trial.suggest_discrete_uniform('catnb_alpha', 0.0, 1.0, 0.05)
        fit_prior            = trial.suggest_categorical('catnb_fit_prior', [False, True])
        feature_scaler       = MinMaxScaler()
        classifier_tmp       = CategoricalNB(alpha=alpha,
                                             fit_prior=fit_prior)
        classifier_obj       = make_pipeline(feature_scaler, classifier_tmp)
    
    elif classifier_name    == 'CDClassifier':
        loss                 = trial.suggest_categorical('cdc_loss', ['squared_hinge', 'log', 'modified_huber', 'squared'])
        penalty              = trial.suggest_categorical('cdc_penalty', ['l1', 'l2', 'l1/l2'])
        multiclass           = loss in ['squared_hinge', 'log'] and penalty == 'l1/l2'
        C                    = trial.suggest_loguniform('cdc_C', 1e-3, 1e3)
        alpha                = trial.suggest_loguniform('cdc_alpha', 1e-3, 1e3)
        selection            = trial.suggest_categorical('cdc_selection', ['cyclic', 'uniform'])
        permute              = selection == 'cyclic'
        classifier_obj       = CDClassifier(loss=loss,
                                            penalty=penalty,
                                            multiclass=multiclass,
                                            C=C,
                                            alpha=alpha,
                                            selection=selection,
                                            permute=permute,
                                            random_state=SEED)

    elif classifier_name    == 'ComplementNB':
        alpha                = trial.suggest_discrete_uniform('compnb_alpha', 0.0, 1.0, 0.05)
        fit_prior            = trial.suggest_categorical('compnb_fit_prior', [False, True])
        norm                 = trial.suggest_categorical('compnb_norm', [False, True])
        classifier_obj       = ComplementNB(alpha=alpha,
                                            fit_prior=fit_prior,
                                            norm=norm)

    elif classifier_name    == 'DecisionTreeClassifier':
        criterion            = trial.suggest_categorical('dtc_criterion', ['gini', 'entropy'])
        splitter             = trial.suggest_categorical('dtc_splitter', ['best', 'random'])
        min_samples_split    = trial.suggest_int('dtc_min_samples_split', 2, 50)
        min_samples_leaf     = trial.suggest_int('dtc_min_samples_leaf', 1, 50)
        max_features         = trial.suggest_int('dtc_max_features', 1, X_train.shape[1])
        classifier_obj       = DecisionTreeClassifier(criterion=criterion, 
                                                      splitter=splitter, 
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      max_features=max_features,
                                                      random_state=SEED)
    

    elif classifier_name    == 'EasyEnsembleClassifier':
        n_estimators         = trial.suggest_int('eec_n_estimators', 5, 20, 1)
        sampling_strategy    = trial.suggest_categorical('eec_sampling_strategy', ['majority', 'not minority', 'not majority', 'all'])
        classifier_obj       = EasyEnsembleClassifier(n_estimators=n_estimators,
                                                      sampling_strategy=sampling_strategy,
                                                      random_state=SEED,
                                                      n_jobs=N_JOBS)

    elif classifier_name    == 'eLCS':
        learning_iterations  = trial.suggest_categorical('elcs_learning_iterations', [2000])
        N                    = trial.suggest_int('elcs_n', 100, 1000, 100)
        p_spec               = trial.suggest_discrete_uniform('elcs_p_spec', 0.0, 1.0, 0.05)
        nu                   = trial.suggest_int('elcs_nu', 1, 5)
        classifier_obj       = eLCS(learning_iterations=learning_iterations,
                                    N=N,
                                    p_spec=p_spec,
                                    nu=nu,
                                    random_state=SEED)
    
    elif classifier_name    == 'ExtraTreeClassifier':
        criterion            = trial.suggest_categorical('etc_criterion', ['gini', 'entropy'])
        min_samples_split    = trial.suggest_int('etc_min_samples_split', 2, 50)
        min_samples_leaf     = trial.suggest_int('etc_min_samples_leaf', 1, 50)
        max_features         = trial.suggest_int('etc_max_features', 1, X_train.shape[1])
        classifier_obj       = ExtraTreeClassifier(criterion=criterion,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf,
                                                   max_features=max_features,
                                                   random_state=SEED)

    elif classifier_name    == 'ExtraTreesClassifier':
        n_estimators         = trial.suggest_int('etsc_n_estimators', 50, 200, 10)
        criterion            = trial.suggest_categorical('etsc_criterion', ['gini', 'entropy'])
        min_samples_split    = trial.suggest_int('etsc_min_samples_split', 2, 50)
        min_samples_leaf     = trial.suggest_int('etsc_min_samples_leaf', 1, 50)
        max_features         = trial.suggest_int('etsc_max_features', 1, X_train.shape[1])
        bootstrap            = trial.suggest_categorical('etsc_bootstrap', [False, True])
        classifier_obj       = ExtraTreesClassifier(n_estimators=n_estimators,
                                                    criterion=criterion, 
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    max_features=max_features,
                                                    bootstrap=bootstrap,
                                                    n_jobs=N_JOBS,
                                                    random_state=SEED)
    
    elif classifier_name    == 'GaussianNB':
        var_smoothing        = trial.suggest_loguniform('gnb_var_smoothing', 1e-12, 1e3)
        classifier_obj       = GaussianNB(var_smoothing=var_smoothing)
    
    elif classifier_name    == 'GaussianProcessClassifier':
        max_iter_predict     = trial.suggest_int('gpc_max_iter_predict', 50, 200, 10)
        multi_class          = trial.suggest_categorical('gpc_multi_class', ['one_vs_one', 'one_vs_rest'])
        classifier_obj       = GaussianProcessClassifier(max_iter_predict=max_iter_predict,
                                                         multi_class=multi_class,
                                                         n_jobs=N_JOBS,
                                                         random_state=SEED)
    
    elif classifier_name    == 'HistGradientBoostingClassifier':
        learning_rate        = trial.suggest_loguniform('hgbc_learning_rate', 1e-3, 1e0)
        max_iter             = trial.suggest_categorical('hgbc_max_iter', [2000])
        min_samples_leaf     = trial.suggest_int('hgbc_min_samples_leaf', 1, 50)
        l2_regularization    = trial.suggest_discrete_uniform('hgbc_l2_regularization', 0.0, 1.0, 0.2)
        max_bins             = trial.suggest_categorical('hgbc_max_bins', [63, 127, 255])
        early_stopping       = trial.suggest_categorical('hgbc_early_stopping', [True])
        validation_fraction  = trial.suggest_categorical('hgbc_validation_fraction', [0.25])
        n_iter_no_change     = trial.suggest_categorical('hgbc_n_iter_no_change', [10])
        classifier_obj       = HistGradientBoostingClassifier(learning_rate=learning_rate,
                                                              max_iter=max_iter,
                                                              min_samples_leaf=min_samples_leaf,
                                                              l2_regularization=l2_regularization,
                                                              max_bins=max_bins,
                                                              early_stopping=early_stopping,
                                                              validation_fraction=validation_fraction,
                                                              n_iter_no_change=n_iter_no_change,
                                                              random_state=SEED)
    
    elif classifier_name    == 'KNeighborsClassifier':
        n_neighbors          = trial.suggest_int('knc_n_neighbors', 1, 15, 2)
        leaf_size            = trial.suggest_int('knc_leaf_size', 10, 100, 10)
        metric               = trial.suggest_categorical('knc_metric', ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'mahalanobis'])
        classifier_obj       = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                    leaf_size=leaf_size,
                                                    metric=metric,
                                                    n_jobs=[N_JOBS])

    elif classifier_name    == 'LGBMClassifier':
        num_leaves           = trial.suggest_categorical('lgbm_num_leaves', [7, 15, 31, 63, 127])
        learning_rate        = trial.suggest_loguniform('lgbm_learning_rate', 1e-6, 1e0)
        n_estimators         = trial.suggest_int('lgbm_n_estimators', 10, 200, 10)
        reg_alpha            = trial.suggest_discrete_uniform('lgbm_reg_alpha', 0.0, 1.0, 0.05)
        reg_lambda           = trial.suggest_discrete_uniform('lgbm_reg_lambda', 0.0, 1.0, 0.05)
        classifier_obj       = LGBMClassifier(num_leaves=num_leaves,
                                              learning_rate=learning_rate,
                                              n_estimators=n_estimators,
                                              reg_alpha=reg_alpha,
                                              reg_lambda=reg_lambda,
                                              n_jobs=N_JOBS,
                                              random_state=SEED)
    
    elif classifier_name    == 'LinearDiscriminantAnalysis':
        solver               = trial.suggest_categorical('lda_solver', ['svd', 'lsqr', 'eigen'])
        shrinkage            = None if solver == 'svd' else 'auto'
        n_features,n_classes = X_train.shape[1],len(set(y_train))
        n_components         = trial.suggest_int('lda_n_components', 1, min(n_features, n_classes-1), 1)
        tol                  = 1e-4 if solver == 'svd' else trial.suggest_categorical('lda_tol', [1e-5, 1e-4, 1e-3])
        classifier_obj       = LinearDiscriminantAnalysis(solver=solver,
                                                          shrinkage=shrinkage,
                                                          n_components=n_components,
                                                          tol=tol)
    
    elif classifier_name    == 'LinearSVC':
        dual                 = trial.suggest_categorical('lsvc_dual', [False])
        C                    = trial.suggest_loguniform('lsvc_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('lsvc_max_iter', [2000])
        classifier_obj       = LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
    
    elif classifier_name    == 'LinearSVC_AdditiveChi2Sampler':
        kernel_sample_steps  = trial.suggest_categorical('lsvc_ac2s_kernel_sample_steps', [1, 2, 3])
        dual                 = trial.suggest_categorical('lsvc_ac2s_dual', [False])
        C                    = trial.suggest_loguniform('lsvc_ac2s_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('lsvc_ac2s_max_iter', [2000])
        feature_mapper       = AdditiveChi2Sampler(sample_steps=kernel_sample_steps)
        classifier_tmp       = LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'LinearSVC_Nystroem':
        kernel_gamma         = trial.suggest_loguniform('lsvc_nystroem_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('lsvc_nystroem_kernel_n_components', 50, 150, 50)
        dual                 = trial.suggest_categorical('lsvc_nystroem_dual', [False])
        C                    = trial.suggest_loguniform('lsvc_nystroem_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('lsvc_nystroem_max_iter', [2000])
        feature_mapper       = Nystroem(gamma=kernel_gamma,
                                        n_components=kernel_n_components,
                                        random_state=SEED)
        classifier_tmp       = LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'LinearSVC_PolynomialCountSketch':
        kernel_gamma         = trial.suggest_loguniform('lsvc_pcs_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('lsvc_pcs_kernel_n_components', 50, 150, 50)
        kernel_degree        = trial.suggest_categorical('lsvc_pcs_degree', [2])
        dual                 = trial.suggest_categorical('lsvc_pcs_dual', [False])
        C                    = trial.suggest_loguniform('lsvc_pcs_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('lsvc_pcs_max_iter', [2000])
        feature_mapper       = PolynomialCountSketch(gamma=kernel_gamma,
                                                     n_components=kernel_n_components,
                                                     degree=kernel_degree,
                                                     random_state=SEED)
        classifier_tmp       = LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'LinearSVC_RBFSampler':
        kernel_gamma         = trial.suggest_loguniform('lsvc_rbfs_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('lsvc_rbfs_kernel_n_components', 50, 150, 50)
        dual                 = trial.suggest_categorical('lsvc_rbfs_dual', [False])
        C                    = trial.suggest_loguniform('lsvc_rbfs_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('lsvc_rbfs_max_iter', [2000])
        feature_mapper       = RBFSampler(gamma=kernel_gamma,
                                          n_components=kernel_n_components,
                                          random_state=SEED)
        classifier_tmp       = LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'LogisticRegression':
        C                    = trial.suggest_loguniform('lr_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('lr_max_iter', [2000])
        l1_ratio             = trial.suggest_discrete_uniform('lr_l1_ratio', 0.0, 1.0, 0.05)
        classifier_obj       = LogisticRegression(penalty='elasticnet',
                                                  C=C,
                                                  max_iter=max_iter,
                                                  solver='saga',
                                                  n_jobs=N_JOBS,
                                                  l1_ratio=l1_ratio)

    elif classifier_name    == 'MLPClassifier':
        create_hidden_layers = lambda value,count : tuple([int(value*2**(count-i-1)) for i in range(0,count)])
        hidden_layer_count   = trial.suggest_int('mlpc_hidden_layer_count', 1, 3, 1)
        hidden_layer_sizes   = create_hidden_layers(2*X_train.shape[1], hidden_layer_count)
        learning_rate        = trial.suggest_categorical('mlpc_learning_rate', ['constant', 'invscaling', 'adaptive'])
        learning_rate_init   = trial.suggest_loguniform('mlpc_learning_rate_init', 1e-6, 1e0)
        max_iter             = trial.suggest_categorical('mplc_max_iter', [2000])
        classifier_obj       = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                             learning_rate=learning_rate,
                                             learning_rate_init=learning_rate_init,
                                             max_iter=max_iter,
                                             early_stopping=True,
                                             random_state=SEED)

    elif classifier_name    == 'MultinomialNB':
        alpha                = trial.suggest_discrete_uniform('mnb_alpha', 0.0, 1.0, 0.05)
        fit_prior            = trial.suggest_categorical('mnb_fit_prior', [False, True])
        classifier_obj       = MultinomialNB(alpha=alpha,
                                             fit_prior=fit_prior)
    
    elif classifier_name    == 'NearestCentroid':
        metric               = trial.suggest_categorical('nc_metric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
        shrink_threshold     = trial.suggest_categorical('nc_shrink_threshold', [None, 0.2])
        classifier_obj       = NearestCentroid(metric=metric,
                                               shrink_threshold=shrink_threshold)

    elif classifier_name    == 'NuSVC_Linear':
        nu                   = trial.suggest_loguniform('nusvclin_nu', 1e-3, 1e+3)
        classifier_obj       = NuSVC(kernel='linear',
                                     nu=nu,
                                     random_state=SEED)
    
    elif classifier_name    == 'NuSVC_Poly':
        nu                   = trial.suggest_loguniform('nusvcpoly_nu', 1e-3, 1e+3)
        degree               = trial.suggest_categorical('nuscvpoly_degree'), [2, 3, 4]
        gamma                = trial.suggest_categorical('nusvcpoly_gamma', ['auto', 'scale'])
        coef0                = trial.suggest_discrete_uniform('nusvcpoly_coef0', 0.0, 1.0, 0.05)
        classifier_obj       = NuSVC(kernel='poly',
                                     nu=nu,                   
                                     degree=degree,
                                     gamma=gamma,
                                     coef0=coef0,
                                     random_state=SEED)

    elif classifier_name    == 'NuSVC_RBF':
        nu                   = trial.suggest_loguniform('nusvcrbf_nu', 1e-3, 1e+3)
        gamma                = trial.suggest_categorical('nusvcrbf_gamma', ['auto', 'scale'])
        classifier_obj       = NuSVC(kernel='rbf',
                                     nu=nu,
                                     gamma=gamma,
                                     random_state=SEED)
    
    elif classifier_name    == 'NuSVC_Sigmoid':
        nu                   = trial.suggest_loguniform('nusvcsig_nu', 1e-3, 1e+3)
        gamma                = trial.suggest_categorical('nusvcsig_gamma', ['auto', 'scale'])
        coef0                = trial.suggest_discrete_uniform('nusvcsig_coef0', 0.0, 1.0, 0.05)
        classifier_obj       = NuSVC(kernel='sigmoid',
                                     nu=nu,
                                     gamma=gamma,
                                     coef0=coef0,
                                     random_state=SEED)
    
    elif classifier_name    == 'PassiveAggressiveClassifier':
        C                    = trial.suggest_loguniform('pac_C', 1e-6, 1e3)
        early_stopping       = trial.suggest_categorical('pac_early_stopping', [True])
        validation_fraction  = trial.suggest_categorical('pac_validation_fraction', [0.25])
        n_iter_no_change     = trial.suggest_categorical('pac_n_iter_no_change', [10])
        classifier_obj       = PassiveAggressiveClassifier(C=C,
                                                           early_stopping=early_stopping,
                                                           validation_fraction=validation_fraction,
                                                           n_iter_no_change=n_iter_no_change,
                                                           n_jobs=N_JOBS,
                                                           random_state=SEED)
    
    elif classifier_name    == 'Perceptron':
        penalty              = trial.suggest_categorical('p_penalty', ['elasticnet'])
        alpha                = trial.suggest_loguniform('p_alpha', 1e-6, 1e0)
        eta0                 = trial.suggest_discrete_uniform('p_eta0', 0.5, 2.0, 0.5)
        early_stopping       = trial.suggest_categorical('p_early_stopping', [True])
        validation_fraction  = trial.suggest_categorical('p_validation_fraction', [0.25])
        n_iter_no_change     = trial.suggest_categorical('p_n_iter_no_change', [10])
        classifier_obj       = Perceptron(penalty=penalty,
                                          alpha=alpha,
                                          eta0=eta0,
                                          early_stopping=early_stopping,
                                          n_iter_no_change=n_iter_no_change,
                                          n_jobs=N_JOBS,
                                          random_state=SEED)

    elif classifier_name    == 'QuadraticDiscriminantAnalysis':
        reg_param            = trial.suggest_discrete_uniform('qda_reg_param', 0.0, 1.0, 0.05)
        tol                  = trial.suggest_loguniform('qda_tol', 1e-6, 1e-3)
        feature_scaler       = StandardScaler()
        classifier_tmp       = QuadraticDiscriminantAnalysis(reg_param=reg_param,
                                                             tol=tol)
        classifier_obj       = make_pipeline(feature_scaler, classifier_tmp)

    elif classifier_name    == 'RadiusNeighborsClassifier':
        radius               = trial.suggest_categorical('rnc_radius', [0.5, 1.0, 2.0])
        weights              = trial.suggest_categorical('rnc_weights', ['uniform', 'distance'])
        classifier_obj       = RadiusNeighborsClassifier(radius=radius,
                                                         weights=weights,
                                                         n_jobs=N_JOBS,
                                                         random_state=SEED)

    elif classifier_name    == 'RandomForestClassifier':
        criterion            = trial.suggest_categorical('rfc_criterion', ['gini', 'entropy'])
        n_estimators         = trial.suggest_int('rfc_n_estimators', 10, 200, 10)
        min_samples_split    = trial.suggest_int('rfc_min_samples_split', 2, 50)
        min_samples_leaf     = trial.suggest_int('rfc_min_samples_leaf', 1, 50)
        max_features         = trial.suggest_int('rfc_max_features', 1, X_train.shape[1])
        bootstrap            = trial.suggest_categorical('rfc_bootstrap', [False, True])
        classifier_obj       = RandomForestClassifier(n_estimators=n_estimators,
                                                      criterion=criterion, 
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      max_features=max_features,
                                                      bootstrap=bootstrap,
                                                      n_jobs=N_JOBS,
                                                      random_state=SEED)
                                                    
    elif classifier_name    == 'RidgeClassifier':
        alpha                = trial.suggest_loguniform('rc_C', 1e-6, 1e6)
        tol                  = trial.suggest_loguniform('rc_tol', 1e-4, 1e-2)
        classifier_obj       = RidgeClassifier(alpha=alpha,
                                               tol=tol,
                                               random_state=SEED)
    
    elif classifier_name    == 'RUSBoostClassifier':
        n_estimators         = trial.suggest_int('rusbc_n_estimators', 25, 100, 25)
        learning_rate        = trial.suggest_loguniform('rusbc_learning_rate', 1e-6, 1e0)
        sampling_strategy    = trial.suggest_categorical('rusbc_sampling_strategy', ['majority', 'not minority', 'not majority', 'all'])
        classifier_obj       = RUSBoostClassifier(n_estimators=n_estimators,
                                                  learning_rate=learning_rate,
                                                  sampling_strategy=sampling_strategy,
                                                  random_state=SEED) 
    
    elif classifier_name    == 'SGDClassifier':
        loss                 = trial.suggest_categorical('sgdc_loss', ['hinge', 'log', 'modified_huber', 'squared_hinge'])
        penalty              = trial.suggest_categorical('sgdc_penalty', ['elasticnet'])
        alpha                = trial.suggest_loguniform('sgdc_alpha', 1e-6, 1e3)
        l1_ratio             = trial.suggest_discrete_uniform('sgdc_l1_ratio', 0.0, 1.0, 0.2)
        learning_rate        = trial.suggest_categorical('sgdc_learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
        eta0                 = trial.suggest_loguniform('sgdc_eta0', 1e-3, 1e3)
        early_stopping       = trial.suggest_categorical('sgdc_early_stopping', [True])
        validation_fraction  = trial.suggest_categorical('sgdc_validation_fraction', [0.25])
        n_iter_no_change     = trial.suggest_categorical('sgdc_n_iter_no_change', [10])
        classifier_obj       = SGDClassifier(loss=loss,
                                             penalty=penalty,
                                             alpha=alpha,
                                             l1_ratio=l1_ratio,
                                             learning_rate=learning_rate,
                                             eta0=eta0,
                                             early_stopping=early_stopping,
                                             validation_fraction=validation_fraction,
                                             n_iter_no_change=n_iter_no_change,
                                             n_jobs=N_JOBS,
                                             random_state=SEED)
    
    elif classifier_name    == 'TabNetClassifier':
        n_d                  = trial.suggest_categorical('tnc_n_d', [8, 16, 32, 64])
        n_a                  = trial.suggest_categorical('tnc_n_a', [8, 16, 32, 64])
        n_steps              = trial.suggest_int('tnc_n_steps', 3, 10)
        gamma                = trial.suggest_discrete_uniform('tnc_gamma', 1.0, 2.0, 0.1)
        n_independent        = trial.suggest_int('tnc_n_independents', 1, 5)
        n_shared             = trial.suggest_int('tnc_n_shared', 1, 5)
        momentum             = trial.suggest_loguniform('tnc_momentum', 0.01, 0.4)
        lambda_sparse        = trial.suggest_loguniform('tnc_lambda_sparse', 1e-6, 1e0)
        classifier_obj       = TabNetClassifier(n_d=n_d,
                                                n_a=n_a,
                                                n_steps=n_steps,
                                                gamma=gamma,
                                                n_independent=n_independent,
                                                n_shared=n_shared,
                                                momentum=momentum,
                                                lambda_sparse=lambda_sparse)
    
    elif classifier_name    == 'XGBClassifier':
        n_estimators         = trial.suggest_int('xgbc_n_estimators', 10, 200, 10)
        use_label_encoder    = trial.suggest_categorical('xgbc_use_label_encoder', [False])
        learning_rate        = trial.suggest_loguniform('xgbc_learning_rate', 1e-6, 1e0)
        booster              = trial.suggest_categorical('xgbc_booster', ['gbtree', 'gblinear', 'dart'])
        gamma                = trial.suggest_loguniform('xgbc_gamma', 1e-6, 1e0)
        classifier_obj       = XGBClassifier(n_estimators=n_estimators, 
                                             use_label_encoder=use_label_encoder,
                                             learning_rate=learning_rate,
                                             booster=booster,
                                             gamma=gamma,
                                             n_jobs=N_JOBS,
                                             random_state=SEED)
    
    return classifier_obj
