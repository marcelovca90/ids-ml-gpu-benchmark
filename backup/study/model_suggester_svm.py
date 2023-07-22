from random import random
from tkinter import N

from sklearnex import patch_sklearn

patch_sklearn(global_patch=True)
import sklearn
from lightning.classification import LinearSVC as LTNG_LinearSVC
from sklearn.kernel_approximation import (AdditiveChi2Sampler, Nystroem,
                                          PolynomialCountSketch, RBFSampler)
from sklearn.pipeline import make_pipeline
from sklearn.svm._classes import LinearSVC as SKL_LinearSVC

SEED        = 10
N_JOBS      = 1

CLASSIFIER_NAMES = [
    'LTNG_LinearSVC',
    'LTNG_LinearSVC_AdditiveChi2Sampler',
    'LTNG_LinearSVC_Nystroem',
    'LTNG_LinearSVC_PolynomialCountSketch',
    'LTNG_LinearSVC_RBFSampler',
    'SKL_LinearSVC',
    'SKL_LinearSVC_AdditiveChi2Sampler',
    'SKL_LinearSVC_Nystroem',
    'SKL_LinearSVC_PolynomialCountSketch',
    'SKL_LinearSVC_RBFSampler',
]

def get_baseline_suggestion(X_train, y_train, classifier_name, trial):
    
    if classifier_name      == 'LTNG_LinearSVC':
        classifier_obj       = LTNG_LinearSVC(random_state=SEED)
    
    elif classifier_name    == 'LTNG_LinearSVC_AdditiveChi2Sampler':
        classifier_obj       = make_pipeline(AdditiveChi2Sampler(), LTNG_LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'LTNG_LinearSVC_Nystroem':
        classifier_obj       = make_pipeline(Nystroem(random_state=SEED), LTNG_LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'LTNG_LinearSVC_PolynomialCountSketch':
        classifier_obj       = make_pipeline(PolynomialCountSketch(random_state=SEED), LTNG_LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'LTNG_LinearSVC_RBFSampler':
        classifier_obj       = make_pipeline(RBFSampler(random_state=SEED), LTNG_LinearSVC(random_state=SEED))

    if classifier_name      == 'SKL_LinearSVC':
        classifier_obj       = SKL_LinearSVC(random_state=SEED)
    
    elif classifier_name    == 'SKL_LinearSVC_AdditiveChi2Sampler':
        classifier_obj       = make_pipeline(AdditiveChi2Sampler(), SKL_LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'SKL_LinearSVC_Nystroem':
        classifier_obj       = make_pipeline(Nystroem(random_state=SEED), SKL_LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'SKL_LinearSVC_PolynomialCountSketch':
        classifier_obj       = make_pipeline(PolynomialCountSketch(random_state=SEED), SKL_LinearSVC(random_state=SEED))
    
    elif classifier_name    == 'SKL_LinearSVC_RBFSampler':
        classifier_obj       = make_pipeline(RBFSampler(random_state=SEED), SKL_LinearSVC(random_state=SEED))


    return classifier_obj


def get_optimized_suggestion(X_train, y_train, classifier_name, trial):

    if classifier_name    == 'LTNG_LinearSVC':
        loss                 = trial.suggest_categorical('ltng_lsvc_loss', ['hinge', 'squared_hinge'])
        criterion            = trial.suggest_categorical('ltng_lsvc_criterion', ['accuracy', 'auc'])
        C                    = trial.suggest_loguniform('ltng_lsvc_C', 1e-6, 1e3)
        permute              = trial.suggest_categorical('ltng_lsvc_permute', [True, False])
        shrinking            = trial.suggest_categorical('ltng_lsvc_shrinking', [True, False])
        max_iter             = trial.suggest_categorical('ltng_lsvc_max_iter', [2000])
        classifier_obj       = LTNG_LinearSVC(loss=loss,
                                              criterion=criterion,
                                              C=C,
                                              permute=permute,
                                              shrinking=shrinking,
                                              max_iter=max_iter,
                                              random_state=SEED)
    
    elif classifier_name    == 'LTNG_LinearSVC_AdditiveChi2Sampler':
        kernel_sample_steps  = trial.suggest_categorical('lsvc_ac2s_kernel_sample_steps', [1, 2, 3])
        loss                 = trial.suggest_categorical('ltng_ac2s_loss', ['hinge', 'squared_hinge'])
        criterion            = trial.suggest_categorical('ltng_ac2s_criterion', ['accuracy', 'auc'])
        C                    = trial.suggest_loguniform('ltng_ac2s_C', 1e-6, 1e3)
        permute              = trial.suggest_categorical('ltng_ac2s_permute', [True, False])
        shrinking            = trial.suggest_categorical('ltng_ac2s_shrinking', [True, False])
        max_iter             = trial.suggest_categorical('ltng_ac2s_max_iter', [2000])
        feature_mapper       = AdditiveChi2Sampler(sample_steps=kernel_sample_steps)
        classifier_tmp       = LTNG_LinearSVC(loss=loss,
                                              criterion=criterion,
                                              C=C,
                                              permute=permute,
                                              shrinking=shrinking,
                                              max_iter=max_iter,
                                              random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'LTNG_LinearSVC_Nystroem':
        kernel_gamma         = trial.suggest_loguniform('ltng_nystroem_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('ltng_nystroem_kernel_n_components', 50, 150, 50)
        loss                 = trial.suggest_categorical('ltng_nystroem_loss', ['hinge', 'squared_hinge'])
        criterion            = trial.suggest_categorical('ltng_nystroem_criterion', ['accuracy', 'auc'])
        C                    = trial.suggest_loguniform('ltng_nystroem_C', 1e-6, 1e3)
        permute              = trial.suggest_categorical('ltng_nystroem_permute', [True, False])
        shrinking            = trial.suggest_categorical('ltng_nystroem_shrinking', [True, False])
        max_iter             = trial.suggest_categorical('ltng_nystroem_max_iter', [2000])
        feature_mapper       = Nystroem(gamma=kernel_gamma,
                                        n_components=kernel_n_components,
                                        random_state=SEED)
        classifier_tmp       = LTNG_LinearSVC(loss=loss,
                                              criterion=criterion,
                                              C=C,
                                              permute=permute,
                                              shrinking=shrinking,
                                              max_iter=max_iter,
                                              random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'LTNG_LinearSVC_PolynomialCountSketch':
        kernel_gamma         = trial.suggest_loguniform('ltng_pcs_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('ltng_pcs_kernel_n_components', 50, 150, 50)
        kernel_degree        = trial.suggest_categorical('ltng_pcs_degree', [2])
        loss                 = trial.suggest_categorical('ltng_pcs_loss', ['hinge', 'squared_hinge'])
        criterion            = trial.suggest_categorical('ltng_pcs_criterion', ['accuracy', 'auc'])
        C                    = trial.suggest_loguniform('ltng_pcs_C', 1e-6, 1e3)
        permute              = trial.suggest_categorical('ltng_pcs_permute', [True, False])
        shrinking            = trial.suggest_categorical('ltng_pcs_shrinking', [True, False])
        max_iter             = trial.suggest_categorical('ltng_pcs_max_iter', [2000])
        feature_mapper       = PolynomialCountSketch(gamma=kernel_gamma,
                                                     n_components=kernel_n_components,
                                                     degree=kernel_degree,
                                                     random_state=SEED)
        classifier_tmp       = LTNG_LinearSVC(loss=loss,
                                              criterion=criterion,
                                              C=C,
                                              permute=permute,
                                              shrinking=shrinking,
                                              max_iter=max_iter,
                                              random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'LTNG_LinearSVC_RBFSampler':
        kernel_gamma         = trial.suggest_loguniform('ltng_rbfs_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('ltng_rbfs_kernel_n_components', 50, 150, 50)
        loss                 = trial.suggest_categorical('ltng_rbfs_loss', ['hinge', 'squared_hinge'])
        criterion            = trial.suggest_categorical('ltng_rbfs_criterion', ['accuracy', 'auc'])
        C                    = trial.suggest_loguniform('ltng_rbfs_C', 1e-6, 1e3)
        permute              = trial.suggest_categorical('ltng_rbfs_permute', [True, False])
        shrinking            = trial.suggest_categorical('ltng_rbfs_shrinking', [True, False])
        max_iter             = trial.suggest_categorical('ltng_rbfs_max_iter', [2000])
        feature_mapper       = RBFSampler(gamma=kernel_gamma,
                                          n_components=kernel_n_components,
                                          random_state=SEED)
        classifier_tmp       = LTNG_LinearSVC(loss=loss,
                                              criterion=criterion,
                                              C=C,
                                              permute=permute,
                                              shrinking=shrinking,
                                              max_iter=max_iter,
                                              random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)

    elif classifier_name    == 'SKL_LinearSVC':
        dual                 = trial.suggest_categorical('skl_lsvc_dual', [False])
        C                    = trial.suggest_loguniform('skl_lsvc_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('skl_lsvc_max_iter', [2000])
        classifier_obj       = SKL_LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
    
    elif classifier_name    == 'SKL_LinearSVC_AdditiveChi2Sampler':
        kernel_sample_steps  = trial.suggest_categorical('skl_lsvc_ac2s_kernel_sample_steps', [1, 2, 3])
        dual                 = trial.suggest_categorical('skl_lsvc_ac2s_dual', [False])
        C                    = trial.suggest_loguniform('skl_lsvc_ac2s_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('skl_lsvc_ac2s_max_iter', [2000])
        feature_mapper       = AdditiveChi2Sampler(sample_steps=kernel_sample_steps)
        classifier_tmp       = SKL_LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'SKL_LinearSVC_Nystroem':
        kernel_gamma         = trial.suggest_loguniform('skl_lsvc_nystroem_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('skl_lsvc_nystroem_kernel_n_components', 50, 150, 50)
        dual                 = trial.suggest_categorical('skl_lsvc_nystroem_dual', [False])
        C                    = trial.suggest_loguniform('skl_lsvc_nystroem_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('skl_lsvc_nystroem_max_iter', [2000])
        feature_mapper       = Nystroem(gamma=kernel_gamma,
                                        n_components=kernel_n_components,
                                        random_state=SEED)
        classifier_tmp       = SKL_LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'SKL_LinearSVC_PolynomialCountSketch':
        kernel_gamma         = trial.suggest_loguniform('skl_lsvc_pcs_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('skl_lsvc_pcs_kernel_n_components', 50, 150, 50)
        kernel_degree        = trial.suggest_categorical('skl_lsvc_pcs_degree', [2])
        dual                 = trial.suggest_categorical('skl_lsvc_pcs_dual', [False])
        C                    = trial.suggest_loguniform('skl_lsvc_pcs_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('skl_lsvc_pcs_max_iter', [2000])
        feature_mapper       = PolynomialCountSketch(gamma=kernel_gamma,
                                                     n_components=kernel_n_components,
                                                     degree=kernel_degree,
                                                     random_state=SEED)
        classifier_tmp       = SKL_LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    elif classifier_name    == 'SKL_LinearSVC_RBFSampler':
        kernel_gamma         = trial.suggest_loguniform('skl_lsvc_rbfs_kernel_gamma', 1e-3, 1e3)
        kernel_n_components  = trial.suggest_int('skl_lsvc_rbfs_kernel_n_components', 50, 150, 50)
        dual                 = trial.suggest_categorical('skl_lsvc_rbfs_dual', [False])
        C                    = trial.suggest_loguniform('skl_lsvc_rbfs_C', 1e-6, 1e3)
        max_iter             = trial.suggest_categorical('skl_lsvc_rbfs_max_iter', [2000])
        feature_mapper       = RBFSampler(gamma=kernel_gamma,
                                          n_components=kernel_n_components,
                                          random_state=SEED)
        classifier_tmp       = SKL_LinearSVC(dual=dual,
                                         C=C,
                                         max_iter=max_iter,
                                         random_state=SEED)
        classifier_obj       = make_pipeline(feature_mapper, classifier_tmp)
    
    return classifier_obj
