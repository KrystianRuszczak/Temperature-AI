import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import os

path_to_data = "../data/processed/segmented_data.h5"
path_train_data = "../data/processed/train_data.h5"
path_test_data = "../data/processed/test_data.h5"
path_val_data = "../data/processed/val_data.h5"

if os.path.exists(path_train_data and path_test_data and path_val_data):
    print("Data already splitted")
else:
    print("Dividing data into sets")

    with h5py.File(path_to_data, "r") as f:
        X = f["X"][:]
        y = f["y"][:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    def save_to_h5(filename, X_data, y_data):
        with h5py.File(filename, "w") as f:
            f.create_dataset("X", data=X_data, compression="gzip")
            f.create_dataset("y", data=y_data, compression="gzip")

    save_to_h5(path_train_data, X_train, y_train)
    save_to_h5(path_test_data, X_test, y_test)
    save_to_h5(path_val_data, X_val, y_val)

    print("Successfully created data sets")