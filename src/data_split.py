import numpy as np
from sklearn.model_selection import train_test_split
import os

path_to_data = "../data/processed/segmented_data.npz"
with np.load(path_to_data) as data:
    X = data["X"]
    y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

np.savez_compressed("../data/processed/train_data.npz", X=X_train, y=y_train)
np.savez_compressed("../data/processed/test_data.npz", X=X_test, y=y_test)
np.savez_compressed("../data/processed/val_data.npz", X=X_val, y=y_val)

print("Successfully created data sets")