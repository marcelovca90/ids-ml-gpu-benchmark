from pytictoc import TicToc
from numpy import cumsum, min, max
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import legend, plot, xlabel, ylabel, ylim
t = TicToc()
n_samples = 1_000_000
X = df_c.drop(columns=['label']).sample(n_samples, random_state=42)
X = StandardScaler().set_output(transform="pandas").fit_transform(X)
t.tic()
pca = IncrementalPCA(batch_size=10*len(X.columns)).fit(X)
t.toc()
pca_var = cumsum(pca.explained_variance_ratio_)
plot(pca_var, linewidth=2, label='pca')
legend()
xlabel(f'Components (total={len(pca_var)})')
ylabel(f'Explained Variance Ratio (max={max(pca_var):.3f})')

for i in range(len(pca_var)):
    print(f"batch_size={pca.batch_size_}, features={i+1}, cumulative_variance={pca_var[i]}")

# batch_size=6000, features=1, cumulative_variance=0.12699368822520465
# batch_size=6000, features=2, cumulative_variance=0.20868904521681503
# batch_size=6000, features=3, cumulative_variance=0.26918984843511046
# batch_size=6000, features=4, cumulative_variance=0.30641017826744676
# batch_size=6000, features=5, cumulative_variance=0.34344750175499134
# batch_size=6000, features=6, cumulative_variance=0.3795198155052393
# batch_size=6000, features=7, cumulative_variance=0.4124695716946296
# batch_size=6000, features=8, cumulative_variance=0.44154776333230444
# batch_size=6000, features=9, cumulative_variance=0.4692671110135551
# batch_size=6000, features=10, cumulative_variance=0.4924317929253362
# batch_size=6000, features=11, cumulative_variance=0.5135169262965661
# batch_size=6000, features=12, cumulative_variance=0.5329556929684709
# batch_size=6000, features=13, cumulative_variance=0.5520678758168744
# batch_size=6000, features=14, cumulative_variance=0.5709055068619123
# batch_size=6000, features=15, cumulative_variance=0.589540315837118
# batch_size=6000, features=16, cumulative_variance=0.6081505256115276
# batch_size=6000, features=17, cumulative_variance=0.6267198314158757
# batch_size=6000, features=18, cumulative_variance=0.6452738553176645
# batch_size=6000, features=19, cumulative_variance=0.6637931858071138
# batch_size=6000, features=20, cumulative_variance=0.6823119356766039
# batch_size=6000, features=21, cumulative_variance=0.7008304993321073
# batch_size=6000, features=22, cumulative_variance=0.7193490343526908
# batch_size=6000, features=23, cumulative_variance=0.7378675534753896
# batch_size=6000, features=24, cumulative_variance=0.7563860700164384
# batch_size=6000, features=25, cumulative_variance=0.7749045562740526
# batch_size=6000, features=26, cumulative_variance=0.7934229691850064
# batch_size=6000, features=27, cumulative_variance=0.8119061730371738
# batch_size=6000, features=28, cumulative_variance=0.8303468267071137
# batch_size=6000, features=29, cumulative_variance=0.8487372938424605
# batch_size=6000, features=30, cumulative_variance=0.8669653778436357
# batch_size=6000, features=31, cumulative_variance=0.8850200595941928
# batch_size=6000, features=32, cumulative_variance=0.9026752282105756
# batch_size=6000, features=33, cumulative_variance=0.9198734997006606
# batch_size=6000, features=34, cumulative_variance=0.9362170512023313
# batch_size=6000, features=35, cumulative_variance=0.949492008992927
# batch_size=6000, features=36, cumulative_variance=0.9604392570138216
# batch_size=6000, features=37, cumulative_variance=0.9694347920289672
# batch_size=6000, features=38, cumulative_variance=0.9776036810451546
# batch_size=6000, features=39, cumulative_variance=0.9842926918172353
# batch_size=6000, features=40, cumulative_variance=0.9899303576592056
# batch_size=6000, features=41, cumulative_variance=0.9935446402516424
# batch_size=6000, features=42, cumulative_variance=0.9969587039149265
# batch_size=6000, features=43, cumulative_variance=0.9991081804644635
# batch_size=6000, features=44, cumulative_variance=0.9998737381195514
# batch_size=6000, features=45, cumulative_variance=0.9999981708054165
# batch_size=6000, features=46, cumulative_variance=0.9999999989827507
# batch_size=6000, features=47, cumulative_variance=0.9999999999998025
# batch_size=6000, features=48, cumulative_variance=0.9999999999998025
# batch_size=6000, features=49, cumulative_variance=0.9999999999998025
# batch_size=6000, features=50, cumulative_variance=0.9999999999998025
# batch_size=6000, features=51, cumulative_variance=0.9999999999998025
# batch_size=6000, features=52, cumulative_variance=0.9999999999998025
# batch_size=6000, features=53, cumulative_variance=0.9999999999998025
# batch_size=6000, features=54, cumulative_variance=0.9999999999998025
# batch_size=6000, features=55, cumulative_variance=0.9999999999998025
# batch_size=6000, features=56, cumulative_variance=0.9999999999998025
# batch_size=6000, features=57, cumulative_variance=0.9999999999998025
# batch_size=6000, features=58, cumulative_variance=0.9999999999998025
# batch_size=6000, features=59, cumulative_variance=0.9999999999998025
# batch_size=6000, features=60, cumulative_variance=0.9999999999998025
