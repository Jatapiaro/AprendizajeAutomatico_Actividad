# coding=utf-8
from DT import get_results as dt_get_results
from ReadData import read_data
from NeuralNetwork import get_results as neural_get_results
from sklearn.model_selection import train_test_split
from NearestNeighbors import get_results as neig_get_results
import csv

data,features = read_data()
y = data["Performance"]
x = data[features]
# Random split 60% of the data for training and 40% for test
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.60)


media_de_error = 0


media_de_error += dt_get_results(
    data.copy(),features,X_train,
    X_test, y_train, y_test,x,y)
print ("\n----------")
media_de_error += neural_get_results(
    data.copy(),features,X_train,
    X_test, y_train, y_test,x,y)
print ("\n----------")

a,b,c = neig_get_results(data.copy(),features,X_train,
    X_test, y_train, y_test,x,y)

media_de_error += b
print ("\n----------\n")

#print ("Media de error de los resultados: "+str((media_de_error/3)))


resultado = 0
with open("global_results.csv") as f:
    for row in csv.reader(f):
        calc2 = float(row[0]) - float(row[4])
        print (calc2*calc2)
        resultado+=(calc2*calc2)

print ("Error cuadrático medio entre la columna 1 y la columna 5: "+str(float(resultado*0.5)))

