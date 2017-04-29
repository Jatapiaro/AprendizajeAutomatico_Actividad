# coding=utf-8
import pandas as pd
import os
"""
Read .csv file and return a pandas dataframe
"""
def read_data():
    if os.path.exists("dataset.csv"):
        print("File founded")
        data = pd.read_csv("dataset.csv")
        features = list(data.columns[1:7])
    return data,features