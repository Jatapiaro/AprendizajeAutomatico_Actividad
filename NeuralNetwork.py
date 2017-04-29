# coding=utf-8
from __future__ import print_function
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,confusion_matrix
import csv

from sklearn.neural_network import MLPClassifier



def get_results(data,features,X_train,X_test,y_train,y_test,x,y):

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(6,6,6), max_iter=500)
    print ("Datos MLP: "+str(mlp))
    print ("\-------------/")
    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)
    print("Neural network predictions: "+str(predictions))



    aux = 0
    f = open('neural_results.csv', 'w')
    for i, row in y_test.iteritems():
        #print(str(i) + "," + str(row))
        line = str(row) + "," + str(predictions[aux])
        #print (line)
        aux += 1
        print (line,file = f)
    f.close()

    lines = []
    with open("global_results.csv") as f:
        for row in csv.reader(f):
            calc = str(row[0])+","+str(row[1])
            lines.append(calc)
    aux = 0
    f = open('global_results.csv', 'w')
    for i in range(0,len(lines)):
        line =lines[aux] + ","+ str(predictions[aux])
        aux += 1
        print (line,file = f)
    f.close()




    result = 0
    with open("neural_results.csv") as f:
        total = 0
        for row in csv.reader(f):
            calc = float(row[0]) - float(row[1])
            total += (calc * calc)
        print("Error cuadrático medio(Neural networks): " + str(0.5 * calc))
        print ("\n--------------")
        result = (calc*0.5)
    return result

