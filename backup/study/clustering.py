from pytictoc import TicToc
from sklearn.datasets import load_digits
from matplotlib.pyplot import legend, scatter, plot, xlabel, ylabel, ylim, show
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from imblearn.under_sampling import ClusterCentroids, InstanceHardnessThreshold, TomekLinks
from sklearn.tree import ExtraTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans, AffinityPropagation, SpectralClustering
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

t = TicToc()

# X, y = load_digits(return_X_y=True)

# X_tsne = TSNE().fit_transform(X)
# scatter(X_tsne[:,0], X_tsne[:,1], label='tsne')

# X_pca = PCA(n_components=2).fit_transform(X)
# scatter(X_pca[:,0], X_pca[:,1], label='pca')

tomek_base, tomek_faiss, hardness, cluster_kmeans, cluster_mb_kmeans = [], [], [], [], []

i_range = [i for i in range(1, 10)]

for i in i_range:
    
    n_samples = 10_000 * i
    X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=10, n_classes=3)
    print(i, X.shape, y.shape)

    t.tic()
    XX, yy = TomekLinks(n_jobs=16).fit_resample(X, y)
    tocvalue = t.tocvalue()
    print('\ttomek_base', XX.shape, yy.shape, tocvalue)
    tomek_base.append(tocvalue)

    t.tic()
    XX, yy = TomekLinks(n_jobs=8, faiss=True).fit_resample(X, y)
    tocvalue = t.tocvalue()
    print('\ttomek_faiss', XX.shape, yy.shape, tocvalue)
    tomek_faiss.append(tocvalue)

    # t.tic()
    # XX, yy = InstanceHardnessThreshold(estimator=ExtraTreeClassifier(random_state=42), random_state=42).fit_resample(X, y)
    # tocvalue = t.tocvalue()
    # print('\tinstance_hardness', XX.shape, yy.shape, tocvalue)
    # hardness.append(tocvalue)

    # t.tic()
    # XX, yy = ClusterCentroids(estimator=KMeans(n_clusters=10), random_state=42).fit_resample(X, y)
    # tocvalue = t.tocvalue()
    # print('\tcluster_kmeans', XX.shape, yy.shape, tocvalue)
    # cluster_kmeans.append(tocvalue)

    # t.tic()
    # XX, yy = ClusterCentroids(estimator=MiniBatchKMeans(n_clusters=10, random_state=42), random_state=42).fit_resample(X, y)
    # tocvalue = t.tocvalue()
    # print('\tcluster_mb_kmeans', XX.shape, yy.shape, tocvalue)
    # cluster_mb_kmeans.append(tocvalue)

plot(i_range, tomek_base, label='tomek_base')
plot(i_range, tomek_faiss, label='tomek_faiss')
# plot(i_range, hardness, label='instance_hardness')
# plot(i_range, cluster_kmeans, label='cluster_kmeans')
# plot(i_range, cluster_mb_kmeans, label='cluster_mb_kmeans')
legend()
xlabel(f'n_samples multipliter')
ylabel(f'fit_resample time')
show(block=True)