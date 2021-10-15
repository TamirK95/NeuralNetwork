import sys
import numpy as np
import random
import NeuralNetworkModule

def input_error():
    """
    Print error message and exit in case of invalid input.
    rtype: None
    """
    print("Invalid Input!", end="")
    sys.exit()

def checkFirstLineNotEmpty(first_line):
    """
    Check that the first line in the provided file is not empty.
    type first_line: str
    rtype: None 
    """
    if first_line == "" or first_line is None:
        input_error()

def data_minmax(dataset):
    """
    Find and Return min and max values for each column in dataset.

    type dataset = list[float]
    rtype = list[list[float]]
    """
    minmax = list()
    res = [[min(column), max(column)] for column in zip(*dataset)]
    return res

def normalize_dataset(dataset, minmax):
    """
    Rescale dataset columns to range [0,1].

    type dataset = list[float]
    type minmax = list[list[float]]
    rtype = None
    """
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])

random.seed(42)
#specify file name and data precentage to be taken for testing:
file_name = 'seeds_dataset.csv'
test_ratio = 0.25

#read data:
with open(file_name) as f:
    first_line = f.readline()
f.close()
checkFirstLineNotEmpty(first_line)
first_line_list = first_line.split(",")
row_size = len(first_line_list)
all_data = np.genfromtxt(file_name, delimiter=",")

#pre-processing:
data_list = all_data.tolist()
np.random.shuffle(data_list)
data_list = np.array(data_list)
data_size = data_list.shape[0]
minmax = data_minmax(data_list)
normalize_dataset(data_list, minmax)
train_data = data_list[:int((1-test_ratio)*data_size), :]
test_data = data_list[int((1-test_ratio)*data_size):, :]
test_actual = test_data[:, row_size-1:]
test_actual = np.transpose(test_actual)
test_actual = test_actual[0]
test_data = test_data[:, :row_size-1]

#calling module:
l_rate = 0.3
n_epoch = 500
n_hidden_layers = 1
NeuralNetworkModule.fit(train_data.tolist(), test_data.tolist(), test_actual.tolist(), train_data.shape[0], row_size, test_data.shape[0], len(set([row[-1] for row in train_data])), n_epoch, n_hidden_layers, l_rate)