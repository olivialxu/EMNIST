##############################################
##   EMNIST Hand Lettering Classification   ##
##############################################
#
# Authors: Olivia Xu, Julia Xi, Roshen Nair
# Date: March 7, 2025
#
# ????
# 
# Dataset: www.kaggle.com/competitions/cme-250-win-2025-emnist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import torch
import torch.nn as nn
import torch.optim as optim

def char_to_idx(letter):
    return ord(letter) - ord('a')

def idx_to_char(idx):
    return chr(idx + ord('a'))

def load_data(dataset, testing):
    # add column labels
    letter_name = 'letter'
    feature_names = [f"pxl_{i}" for i in range(784)]

    # load EMNIST letters dataset
    df = pd.read_csv(dataset, header=None, names=[letter_name] + feature_names)

    # split into features and labels
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[letter_name].to_numpy(dtype=np.int64)

    # split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # load test data
    test_df = pd.read_csv(testing, header=None, names=feature_names, skiprows=1)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32)

    # reshape data to 2D
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return df, X_train, y_train, X_val, y_val, X_test

def display_image(img_arr, str_idx):
    plt.imshow(img_arr.reshape(28, 28).T)
    plt.title(idx_to_char(str_idx))
    plt.show()

def reduce_pca(X_train, X_val, X_test, n_components=100):
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.2f}")
    return X_train_pca, X_val_pca, X_test_pca

def logistic_regression(X_train, y_train, X_val, y_val):
    model = LogisticRegression(solver='saga', max_iter=1000, C=0.1)  # L2 regularization
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    letters = [idx_to_char(x) for x in range(0, 26)]

    print("Training Performance:")
    print(classification_report(y_train, train_pred, target_names=letters))

    print("Validation Performance:")
    print(classification_report(y_val, val_pred, target_names=letters))

    return model

if __name__ == "__main__":

    df, X_train, y_train, X_val, y_val, X_test = load_data('data/emnist-letters-train.csv', 'data/features-test.csv')

    # for i in range(10):
    #     display_image(X_train[i], y_train[i])

    # apply PCA reduction
    X_train_pca, X_val_pca, X_test_pca = reduce_pca(X_train, X_val, X_test, n_components=100)

    # train logistic regression model
    print("RUNNING LOGISTIC REGRESSION")
    # model = logistic_regression(X_train_pca, y_train, X_val_pca, y_val)