# coding=utf-8
from __future__ import print_function
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import subprocess
from sklearn.model_selection import train_test_split
import csv


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


def get_results(data,features,X_train,X_test,y_train,y_test,x,y):

    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X_train, y_train)

    predictions = dt.predict(X_test)
    print("Desicion tree predictions: "+str(predictions))

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
        print("Error cuadrático medio(DT): " + str(0.5 * calc))
        print("\n--------------")
        result = (calc*0.5)
    return result