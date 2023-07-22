import matplotlib.pyplot as plt
from imblearn.under_sampling import TomekLinks
filename = "C:\\Users\\marce\\git\\iot-threat-classifier\\datasets\\bot_iot\\generated\\macro\\featurewiz_pca\\bot_iot_macro.ftr"
df_c = pd.read_feather(filename).sample(100_000)
X, y = df_c.drop(columns=['label']), df_c['label']
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
autopct = "%.2f"
y.value_counts().plot.pie(autopct=autopct, ax=axs[0])
axs[0].set_title("Original")
X, y = TomekLinks(sampling_strategy='auto', n_jobs=-1).fit_resample(X, y)
y.value_counts().plot.pie(autopct=autopct, ax=axs[1])
axs[1].set_title("Cleaning")
fig.tight_layout()
fig.show()
fig.savefig(filename.replace('.ftr', '.png'))
