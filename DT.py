# coding=utf-8
from __future__ import print_function
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import subprocess
from sklearn.model_selection import train_test_split
import csv


"""
Read .csv file and return a pandas dataframe
"""
def read_data():
    if os.path.exists("dataset.csv"):
        print("File founded")
        data = pd.read_csv("dataset.csv")
    return data

"""
To enable scikit-learn to use our data we must encode
the "names" (first column) to integers
"""
def preprocessing(data_frame,target_column):
    data = data_frame.copy()
    targets = data[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    data["Target"] = data[target_column].replace(map_to_int)
    return (data, targets)

def visualize_tree(tree, feature_names):
    with open("desicion_tree.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names,
                        filled=True, rounded=True,
                        special_characters=True)
    command = ["dot", "-Tpng", "desicion_tree.dot", "-o", "desicion_tree.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def get_results():
    data = read_data()

    features = list(data.columns[1:7])

    # print (features)

    y = data["Performance"]
    x = data[features]

    # Random split 60% of the data for training and 40% for test
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.60)

    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X_train, y_train)

    predictions = dt.predict(X_test)
    print(predictions)
    # print (y_test)
    # print (data)

    aux = 0
    f = open('dt_results.csv', 'w')
    for i, row in y_test.iteritems():
        line = str(row) + "," + str(predictions[aux])
        aux += 1
        print(line, file=f)
    f.close()

    result = 0

    with open("dt_results.csv") as f:
        total = 0
        for row in csv.reader(f):
            calc = float(row[0]) - float(row[1])
            total += (calc * calc)
        print("Error cuadrático medio: " + str(0.5 * calc))
        result = (calc*0.5)
    return result