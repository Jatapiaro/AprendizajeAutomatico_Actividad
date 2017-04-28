import os
import pandas as pd
from math import floor


def read_data():
    """Get the iris data, from local csv or pandas repo."""
    if os.path.exists("dataset.csv"):
        print("File founded")
        data = pd.read_csv("dataset.csv", index_col=0)
    return data

data = read_data()
data_size = len(data.index)
training_rows = int(floor((data_size*60)/100))
test_rows = data_size - training_rows

