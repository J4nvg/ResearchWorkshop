import numpy as np
import pandas as pd
from sklearn import datasets
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def gen_train_test_data(dataset="", train_size=1.0, test_size=0.2, normalize_x=True, seed=42):
    print("dataset: {}, seed: {}".format(dataset, seed))
    print("train_size: {}, test_size: {}".format(train_size, test_size))

    if dataset == "aug_temp":
        X_train = pd.read_csv("data/aug_temp/synthetic_X.csv")
        y_train = pd.read_csv("data/aug_temp/synthetic_y.csv")["target"]

        X_test = pd.DataFrame()
        y_test = pd.Series([], dtype=float)

        n_train = len(X_train)
        n_test = 0
        n_feature = X_train.shape[1]
        n_class = 0
        names = list(X_train.columns)

        print("Loaded synthetic data from aug_temp:")
        print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))

        return X_train, y_train, X_test, y_test, n_train, n_test, n_feature, n_class, names

    # Classic sklearn datasets (classification)
    if dataset == "iris":
        ds = datasets.load_iris(as_frame=True)
        X, y = ds.data, ds.target
        names = ds.feature_names
    elif dataset == "breast_cancer":
        ds = datasets.load_breast_cancer(as_frame=True)
        X, y = ds.data, ds.target
        names = ds.feature_names
    elif dataset == "california":
        ds = datasets.fetch_california_housing(as_frame=True)
        X, y = ds.data, ds.target
        names = ds.feature_names
    else:
        raise ValueError(f"Dataset '{dataset}' not supported in this simplified version.")

    if normalize_x:
        print("normalize X")
        X = MinMaxScaler().fit_transform(X)
        X = np.around(X, 2)
    else:
        print("don't normalize X")

    y = np.array([float(val) for val in y])
    y = y.reshape(-1, 1)
    X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)

    if train_size == 1:
        X_train, y_train = X_train_, y_train_
    else:
        X_train, _, y_train, _ = train_test_split(X_train_, y_train_, test_size=(1.0 - train_size),
                                                  shuffle=True, random_state=seed)

    n_train, n_test, n_feature, n_class = X_train.shape[0], X_test.shape[0], X_train.shape[1], 0
    print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
    print("X_test: {}, y_test: {}".format(X_test.shape, y_test.shape))
    print("n_train: {}, n_test: {}, n_feature: {}, n_class: {}".format(n_train, n_test, n_feature, n_class))
    print("feature_names: {}".format(names))

    return X_train, y_train, X_test, y_test, n_train, n_test, n_feature, n_class, names
