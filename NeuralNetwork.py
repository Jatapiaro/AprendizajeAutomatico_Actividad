# coding=utf-8
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


from sklearn.neural_network import MLPClassifier



def get_results(features):
    data = pd.read_csv('dataset.csv')
    y = data["Performance"]
    x = data[features]
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.60)

    #Estandarización de la información
    scaler = StandardScaler(X_train)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
    print (mlp)

