# coding=utf-8
from DT import get_results
from ReadData import read_data

data,features = read_data()
print (get_results(data,features))


