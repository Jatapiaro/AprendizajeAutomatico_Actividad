# coding=utf-8
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
import csv

def get_results(data,features,X_train,X_test,y_train,y_test,x,y):

    tres_vecinos = KNeighborsClassifier(n_neighbors=3)
    tres_vecinos.fit(X_train,y_train)
    tres_predicted = tres_vecinos.predict(X_test)
    print ("Predicción con k=3: "+str(tres_predicted))

    aux = 0
    f = open('3neighborgs.csv', 'w')
    for i, row in y_test.iteritems():
        #print(str(i) + "," + str(row))
        line = str(row) + "," + str(tres_predicted[aux])
        #print (line)
        aux += 1
        print (line,file = f)
    f.close()

    resultA = 0
    with open("3neighborgs.csv") as f:
        total = 0
        for row in csv.reader(f):
            calc = float(row[0]) - float(row[1])
            total += (calc * calc)
        print("Error cuadrático medio(K=3): " + str(0.5 * calc))
        print ("\n--------------")
        resultA = (calc*0.5)

    siete_vecinos = KNeighborsClassifier(n_neighbors=7)
    siete_vecinos.fit(X_train,y_train)
    siete_predicted = siete_vecinos.predict(X_test)

    aux = 0
    f = open('7neighborgs.csv', 'w')
    for i, row in y_test.iteritems():
        line = str(row) + "," + str(siete_predicted[aux])
        aux += 1
        print (line,file = f)
    f.close()

    resultC = 0
    with open("7neighborgs.csv") as f:
        total = 0
        for row in csv.reader(f):
            calc = float(row[0]) - float(row[1])
            total += (calc * calc)
        print("Error cuadrático medio(K=7): " + str(0.5 * calc))
        print ("\n--------------")
        resultC = (calc*0.5)

    cinco_vecinos = KNeighborsClassifier(n_neighbors=5)
    cinco_vecinos.fit(X_train,y_train)
    cinco_predicted = cinco_vecinos.predict(X_test)
    print ("Predicción con k=5: "+str(cinco_predicted))
    print("Predicción con k=7: " + str(siete_predicted))

    aux = 0
    f = open('5neighborgs.csv', 'w')
    for i, row in y_test.iteritems():
        line = str(row) + "," + str(cinco_predicted[aux])
        aux += 1
        print (line,file = f)
    f.close()

    resultB = 0
    with open("5neighborgs.csv") as f:
        total = 0
        for row in csv.reader(f):
            calc = float(row[0]) - float(row[1])
            total += (calc * calc)
        print("Error cuadrático medio(K=5): " + str(0.5 * calc))
        print ("\n--------------")
        resultB = (calc*0.5)


    lines = []
    acumulado = []
    with open("global_results.csv") as f:
        for row in csv.reader(f):
            calc = str(row[0])+","+str(row[1])+","+str(row[2])
            calc2 = float(row[1])+float(row[2])
            acumulado.append(calc2)
            lines.append(calc)

    aux = 0
    f = open('global_results.csv', 'w')
    for i in range(0,len(lines)):
        mean = acumulado[aux] + float(cinco_predicted[aux])
        line = lines[aux] + ","+ str(cinco_predicted[aux]) + "," + str((mean/3))
        aux += 1
        print (line,file = f)
    f.close()

    return resultA,resultB,resultC