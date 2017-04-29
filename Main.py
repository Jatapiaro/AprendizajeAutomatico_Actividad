# coding=utf-8
from DT import get_results as dt_get_results
from ReadData import read_data
from NeuralNetwork import get_results as neural_get_results

data,features = read_data()
#print (dt_get_results(data,features))
print (neural_get_results(features))


