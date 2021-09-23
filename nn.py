from random import seed
from random import random
from random import randrange
from csv import reader
from math import exp

def load_csv(file_name):
    """
    Read data from a given csv file.

    type file_name = str
    rtype = list[str]
    """
    dataset = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    """
    Convert str column to float column.

    type dataset = list[str]
    type column = int
    rtype = None
    """
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    """
    Find and return the list of optional outputs.

    type dataset = list[str]
    type column = int
    rtype = list[int]
    """
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def data_minmax(dataset):
    """
    Find and Return min and max values for each column in dataset.

    type dataset = list[float]
    rtype = list[list[float]]
    """
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def normalize_dataset(dataset, minmax):
    """
    Rescale dataset columns to range [0,1].

    type dataset = list[float]
    type minmax = list[list[float]]
    rtype = None
    """
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_validation_split(dataset, n_folds):
    """
    Split the dataset into k folds.

    type dataset = list[float]
    type n_folds = int
    rtype = list[list[float]]
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
	    fold = list()
	    while len(fold) < fold_size:
		    index = randrange(len(dataset_copy))
		    fold.append(dataset_copy.pop(index))
	    dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    """
    Calculate and return the accuracy percentage of the results.

    type actual = list[float]
    type predicted = list[float]
    rtype = float
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def initialize_network(n_inputs, n_hidden, n_outputs):
    """
    Create a new neural network with n_inputs inputs, n_hidden neurons in the hidden layer and n_outputs outputs, and return it.

    type n_inputs: int
    type n_hidden: int
    type n_outputs: int
    rtype: neural_network
    """
    hidden_layer = layer(n_neurons = n_hidden, n_weights = n_inputs)
    output_layer = layer(n_neurons = n_outputs, n_weights = n_hidden)
    layers = [hidden_layer, output_layer]
    network = neural_network(layers)
    return network

def activate(weights, bias, inputs):
    """
    Calculate and return the activations function's value by the given weights, bias and inputs.
    
    type weights: list[float]
    type bias: float
    type inputs: list[float]
    rtype: float
    """
    activation = bias
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    """
    Calculate and return the value of Sigmoid activation function's result for activation value provided.
    
    type activation: float
    rtype: float
    """
    return 1. / (1. + exp(-activation))

def forward_propagate(network, row):
    """
    Implement forward propagation for a row of data from the dataset with the neural netword provided, and return the list of outputs.

    type network: neural_network
    type row: list[float]
    rtype: list[float]
    """
    inputs = row
    layers = network.get_layers()
    for layer in layers:
        new_inputs = []
        neurons = layer.get_neurons()
        for neuron in neurons:
            activation = activate(weights = neuron.get_weights(), bias = neuron.get_bias(), inputs = inputs)
            neuron.set_output(transfer(activation))
            new_inputs.append(neuron.get_output())
        inputs = new_inputs
    return inputs
    
def transfer_derivative(output):
    """
    Given an output value from a neuron, calculate and return it's slope.

    type output: float
    rtype: float
    """
    return output * (1. - output)

def backward_propagate_error(network, expected):
    """
    Implement backward propagation for the neural netword provided, and update the delta (=loss function's value) of each neuron accordingly.

    type network = neural_network
    type expected = list[float]
    rtype = None
    """
    for i in reversed(range(len(network.get_layers()))):
        layers = network.get_layers()
        layer = layers[i]
        errors = list()
        if i != len(network.get_layers()) - 1:   # not output layer
            for j in range(len(layer.get_neurons())):
                error = 0.0
                next_layer = layers[i+1]
                for neuron in next_layer.get_neurons():
                    weights = neuron.get_weights()
                    delta = neuron.get_delta()
                    error += weights[j] * delta
                errors.append(error)
        else:   # output layer
            for j in range(len(layer.get_neurons())):
                neuron = layer.get_neurons()[j]
                errors.append(expected[j] - neuron.get_output())
        for j in range(len(layer.get_neurons())):
            neuron = layer.get_neurons()[j]
            neuron.set_delta(errors[j] * transfer_derivative(neuron.get_output()))

def update_weights(network, row, l_rate):
    """
    Updates the weights for a given network, given an input row of data and learning rate hyperparameter.

    type network: neural_network
    type row: list[float]
    type l_rate: float
    rtype: None
    """
    layers = network.get_layers()
    for i in range(len(layers)):
        inputs = row[:-1]
        if i != 0:  # not first layer
            inputs = [neuron.get_output() for neuron in layers[i-1].get_neurons()]
        neurons = layers[i].get_neurons()
        for neuron in neurons:
            for j in range(len(inputs)):
                neuron.set_weights(j, neuron.get_weights()[j] + l_rate * neuron.get_delta() * inputs[j])
            neuron.set_weights(-1, neuron.get_weights()[-1] + l_rate * neuron.get_delta())

def train_network(network, train, l_rate, n_epoch, n_outputs):
    """
    Implement training of neural network.

    type network = neural_network
    type train = list[float]
    type l_rate: float
    type n_epoch = int
    type n_outputs - int
    rtype = None
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch= ',epoch,' lrate = ',l_rate,' error = ',sum_error)

def predict(network, row):
    """
    Implement procedure of predicting the output by the provided data.

    type network = neural_network
    type row = list[float]
    rtype = int
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def train_and_predict(train, test, l_rate, n_epoch, n_hidden):
    """
    Initialize a model, train it on the test set and predict the test set. Then return the predictions.

    type train = list[float]
    type test = list[float]
    type l_rate = float
    type n_epoch = int
    type n_hidden = int
    rtype = list[int]
    """
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions

class neuron:
    def __init__(self, n_weights):
        self.weights = [random() for i in range(n_weights)]
        self.bias = random()
        self.output = -1
        self.delta = -1
    
    def __repr__(self):
        rep = 'Neuron(weights: ' + str(self.weights) + ', bias: ' + str(self.bias) + ', output: ' + str(self.output) + ', delta: ' + str(self.delta) + ')'
        return rep

    def __str__(self):
        return repr(self)
    
    def get_bias(self):
        return self.bias

    def get_weights(self):
        return self.weights

    def set_output(self, output):
        self.output = output
    
    def get_output(self):
        return self.output
    
    def get_delta(self):
        return self.delta

    def set_delta(self, delta):
        self.delta = delta
    
    def set_weights(self, j, value):
        self.weights[j] = value


class layer:
    def __init__(self, n_neurons, n_weights):
        self.neurons = [neuron(n_weights) for i in range(n_neurons)]
    
    def __repr__(self):
        rep = 'Layer(' + str([neuron for neuron in self.neurons]) + ')\n'
        return rep

    def __str__(self):
        return repr(self)
    
    def get_neurons(self):
        return self.neurons

class neural_network:
    def __init__(self, layers):
        self.layers = layers
    
    def __repr__(self):
        rep = 'network:\n' + str([layer for layer in self.layers])
        return rep

    def __str__(self):
        return repr(self)

    def get_layers(self):
        return self.layers

seed(42)
file_name = 'seeds_dataset.csv'
dataset = load_csv(file_name)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)
minmax = data_minmax(dataset)
normalize_dataset(dataset, minmax)
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, train_and_predict, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: ' , scores)
print('Mean Accuracy: ' , sum(scores) / float(len(scores)))
