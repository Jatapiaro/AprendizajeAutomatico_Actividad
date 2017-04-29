# coding=utf-8
from DT import get_results as dt_get_results
from ReadData import read_data
from NeuralNetwork import get_results as neural_get_results
from sklearn.model_selection import train_test_split

data,features = read_data()
y = data["Performance"]
x = data[features]
# Random split 60% of the data for training and 40% for test
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.60)


media_de_error = 0

media_de_error += dt_get_results(
    data.copy(),features,X_train,
    X_test, y_train, y_test,x,y)

media_de_error += neural_get_results(
    data.copy(),features,X_train,
    X_test, y_train, y_test,x,y)


