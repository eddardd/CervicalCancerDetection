import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imread
from skimage.feature import local_binary_pattern

path_to_images = "/home/eduardo/repos/PapSmearClassification/data/"
df = pd.read_excel(os.path.join(path_to_images, 'new_database_results.xls'))
map_id_cls = {
    id: label for id, label in zip(df['ID'].values, df['Class'].values)
}

X = []
y = []
for id in tqdm(df['ID'].values):
    img = imread(os.path.join(path_to_images, 'database', 'classification', id))
    lbp = [
        local_binary_pattern(img[:, :, i], P=4, R=32) for i in range(3)
    ]
    x = []
    for l in lbp:
        x.append(np.histogram(l.flatten(), bins=np.arange(16))[0])
    x = np.concatenate(x)
    x = x.astype(float) / np.linalg.norm(x)
    X.append(x)
    y.append(map_id_cls[id])
X = np.array(X)
y = np.array(y).reshape(-1, 1)
lbp_dataset = np.concatenate([X, y], axis=1)
np.save(os.path.join(path_to_images, 'lbp_features.npy'), lbp_dataset)