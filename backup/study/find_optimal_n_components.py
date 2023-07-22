import os

import numpy as np
import pandas as pd
from pytictoc import TicToc
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

t = TicToc()

def find_optimal_n_components(df, label_column, variance_threshold):
    X, y = df.drop(columns=[label_column]), df[label_column]
    X_columns, X_transformed = X.columns, StandardScaler().fit_transform(X, y)
    current_n_components = 0
    current_variance_ratio = 0.0
    while current_variance_ratio < variance_threshold:
        t.tic()
        current_n_components += 1
        ipca = IncrementalPCA(n_components=current_n_components, batch_size=100*len(X_columns)).fit(X_transformed)
        current_variance_ratio = np.max(np.cumsum(ipca.explained_variance_ratio_))
        print(f'n_components={current_n_components} -> variance_ratio={current_variance_ratio:.6f} (iteration took {t.tocvalue(True):.2}s)')
    return current_n_components, current_variance_ratio

if __name__ == "__main__":
    full_filename = os.path.join(os.path.dirname(__file__), '../datasets/bot_iot/generated/bot_iot_checkpoint.ftr') 
    df = pd.read_feather(full_filename)
    n_components, variance_ratio = find_optimal_n_components(df, 'label', 0.95)
    print(n_components, variance_ratio)
