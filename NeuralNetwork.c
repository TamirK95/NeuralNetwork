#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

struct neuron
{
    float* weights;
    float bias;
    float output;
    float delta;
    int num_of_weights;
};

struct layer
{
    struct neuron* neurons;
    int num_of_neurons;
};

struct neural_network 
{
    struct layer* layers;
    int num_of_layers;
};

void set_network();
struct neural_network initialize_network(struct neural_network network, int n_inputs, int n_hidden, int n_outputs, int n_hidden_layers);
float activate(float* weights, float bias, float* inputs, int n_weights);
float transfer(float activation);
float* forward_propagate(float* inputs, struct neural_network *network, float* row, int row_size);
float transfer_derivative(float output);
void update_delta(struct neural_network *network, int i, int j, float delta);
void backward_propagate_error(struct neural_network *network, float* expected);
void update_weights(struct neural_network* network, float* row, int row_size, float l_rate);
struct neural_network* train_network(struct neural_network *network, float** train, int train_size, int row_size, float l_rate, int n_epoch, int n_outputs);
int predict(struct neural_network *network, float* row, int row_size, int n_outputs);
float grade(float *actual, int *predicted, int test_dataset_size);
void evaluate_network(struct neural_network* network, float**test_dataset, float* test_actual, int test_dataset_size, int row_size, int n_outputs);
float* get_row(float **dataset, float *row, int i, int row_size);
void copyArr1ToArr2(float *arr1, float *arr2, int n);
void print_network(struct neural_network network);
void print_layer(struct layer layer);
void print_neuron(struct neuron neuron);
void free_network(struct neural_network* network);

struct neural_network initialize_network(struct neural_network network, int n_inputs, int n_hidden, int n_outputs, int n_hidden_layers)
{
    int i, j, t;
    struct layer* layers;
    float* weights;
    network.num_of_layers = n_hidden_layers+1;
    layers = (struct layer *) malloc((n_hidden_layers+1) * (sizeof(struct layer)));
    assert(layers != NULL);
    /*creating the hidden layers*/
    for(i = 0; i < n_hidden_layers; i++)
    {
        struct layer layer;
        struct neuron* neurons = (struct neuron *) malloc(n_hidden * sizeof(struct neuron));
        assert(neurons != NULL);
        for(j = 0; j < n_hidden; j++)
        {
            struct neuron neuron;
            neuron.bias = (float)(double)rand() / (double)((unsigned)RAND_MAX + 1);
            neuron.output = -1;
            neuron.delta = -1;
            int n_weights;
            if (i == 0)
            {
                n_weights = n_inputs;   
            }
            else
            {
                n_weights = n_hidden;
            }
            weights = (float*) malloc(n_weights * sizeof(float));
            neuron.num_of_weights = n_weights;
            for(t = 0; t < n_weights; t++)
            {
                weights[t] = (float)(double)rand() / (double)((unsigned)RAND_MAX + 1);
            }
            neuron.weights = weights;
            neurons[j] = neuron;
        }
        layer.neurons = neurons;
        layer.num_of_neurons = n_hidden;
        layers[i] = layer;
    }
    /*creating the output layer*/
    struct layer output_layer;
    struct neuron* neurons = (struct neuron *) malloc(n_outputs * sizeof(struct neuron));
    assert(neurons != NULL);
    for(j = 0; j < n_outputs; j++)
    {
        struct neuron neuron;
        neuron.bias = (float)(double)rand() / (double)((unsigned)RAND_MAX + 1);
        neuron.output = -1;
        neuron.delta = -1;
        weights = (float*) malloc(n_hidden * sizeof(float));
        neuron.num_of_weights = n_hidden;
        for(t = 0; t < n_hidden; t++)
        {
            weights[t] = (float)(double)rand() / (double)((unsigned)RAND_MAX + 1);
        }
        neuron.weights = weights;
        neurons[j] = neuron;
    }
    output_layer.neurons = neurons;
    output_layer.num_of_neurons = n_outputs;
    layers[n_hidden_layers] = output_layer;
    network.layers = layers;
    return network;
}

/*
* Function: activate
* -----------------------------
* Calculate and return the activations function's value by the given weights, bias and inputs.
*
* weights - a weights of a neuron.
* bias - bias value of a neuron.
* inputs - inputs for a neuron.
*/
float activate(float* weights, float bias, float* inputs, int n_weights)
{
    int i;
    float activation = bias;
    for (i=0; i < n_weights; i++)
    {
        activation += weights[i]*inputs[i];
    }
    return activation;
}

/*
* Function: transfer
* -----------------------------
* Calculate and return the value of Sigmoid activation function's result for activation value provided.
*
* activation - the value of the activation function.
*/
float transfer(float activation)
{
    return 1./(1. + exp((-1.)*activation));
}

/*
* Function: forward_propagate
* -----------------------------
* Implement forward propagation for a row of data from the dataset with the neural netword provided, and return the list of outputs.
*
* network - a given neural network.
* row - a row of data from the dataset.
* row_size - size of row from the dataset.
*/
float* forward_propagate(float* inputs, struct neural_network *network, float* row, int row_size)
{
    int i, j;
    float activation, *new_inputs;
    struct layer *layer, *layers;
    struct neuron *neuron, *neurons;
    copyArr1ToArr2(row, inputs, row_size);
    layers = network->layers;
    for(i=0; i < network->num_of_layers; i++)
    {
        layer = &(layers[i]);
        neurons = layer->neurons;
        new_inputs = (float*) malloc(layer->num_of_neurons * sizeof(float));
        assert(new_inputs != NULL);
        for(j=0; j < layer->num_of_neurons; j++)
        {
            neuron = &neurons[j];
            activation = activate(neuron->weights, neuron->bias, inputs, neuron->num_of_weights);
            neuron->output = transfer(activation);
            new_inputs[j] = neuron->output;
        }
        copyArr1ToArr2(new_inputs, inputs, row_size);
        free(new_inputs);
    }
    return inputs;
}

/*
* Function: transfer_derivative
* -----------------------------
* Given an output value from a neuron, calculate and return it's slope.
*
* output - a given output value of a neuron.
*/
float transfer_derivative(float output)
{
    return (output * (1. - output));
}

/*
* Function: update_delta
* -----------------------------
* Update the delta value of a given network.
*
* network - a pointer to a given neural network.
* i - layer location.
* j - neuron location.
* delta - desired value of delta.
*/
void update_delta(struct neural_network *network, int i, int j, float delta)
{
    struct layer* layers = network->layers;
    layers[i].neurons[j].delta = delta;
    network->layers = layers;
}

/*
* Function: backward_propagate_error
* -----------------------------
* Implement backward propagation for the neural netword provided, and update the delta (=loss function's value) of each neuron accordingly.
*
* network - a pointer to a given neural network.
* expected - expected value (=target).
*/
void backward_propagate_error(struct neural_network* network, float* expected)
{
    int i, j, t, n;
    struct layer layer, next_layer;
    struct neuron neuron;
    float error, *errors;
    n = network->num_of_layers;
    for(i = n-1; i >= 0; i--)
    {
        layer = network->layers[i];
        errors = (float*) malloc(layer.num_of_neurons * sizeof(float));
        assert(errors != NULL);
        /*if not output layer*/
        if (i != n - 1)
        {
            for(j = 0; j < layer.num_of_neurons; j++)
            {
                error = 0.0;
                next_layer = network->layers[i+1];
                for(t=0; t < next_layer.num_of_neurons; t++)
                {
                    neuron = next_layer.neurons[t];
                    error += neuron.weights[j] * neuron.delta;
                }
                errors[j] = error;
            }
        }
        /*output layer:*/
        else
        {
            for(j = 0; j < layer.num_of_neurons; j++)
            {
                neuron = layer.neurons[j];
                errors[j] = expected[j] - neuron.output;
            }
        }
        for(j = 0; j < layer.num_of_neurons; j++)
        {
            neuron = layer.neurons[j];
            update_delta(network, i, j, (errors[j] * transfer_derivative(neuron.output)));
        }
        free(errors);
    }
}

/*
* Function: update_weights
* -----------------------------
* Updates the weights for a given network, given an input row of data and learning rate hyperparameter.
*
* network - a pointer to a given neural network.
* row - row of data points.
* row_size - length of row.
* l_rate - learning rate hyperparameter.
*/
void update_weights(struct neural_network* network, float* row, int row_size, float l_rate)
{
    int i, j, t, n, layer_size, inputs_size;
    float* inputs;
    struct layer layer;
    struct neuron neuron;
    n = network->num_of_layers;
    for(i=0; i<n; i++)
    {
        inputs = (float*) malloc((row_size) * sizeof(float));
        inputs_size = row_size;
        copyArr1ToArr2(row, inputs, row_size);
        if (i != 0)
        {
            layer = network->layers[i-1];
            layer_size = layer.num_of_neurons;
            inputs = (float*) malloc(layer_size * sizeof(float));
            inputs_size = layer_size;
            for(j=0; j < inputs_size; j++)
            {
                inputs[j] = layer.neurons[j].output;
            }
        }
        layer = network->layers[i];
        for(j=0; j<layer.num_of_neurons; j++)
        {
            neuron = layer.neurons[j];
            for(t=0; t<inputs_size; t++)
            {
                neuron.weights[t] += l_rate * neuron.delta * inputs[t];
            }
            neuron.weights[neuron.num_of_weights-1] += l_rate * neuron.delta;
        }
        free(inputs);
    }
}

/*
* Function: train_network
* -----------------------------
* Implement training of neural network.
*
* network - a pointer to a given neural network.
* train - training dataset.
* train_size - number of rows in train dataset.
* row_size - size of a row in the train dataset.
* l_rate - learning rate hyperparameter.
* n_epoch - number of epochs to be done.
* n_outputs - number of possible outputs.
*/
struct neural_network* train_network(struct neural_network *network, float** train, int train_size, int row_size, float l_rate, int n_epoch, int n_outputs)
{
    int i, j, t;
    float sum_error, *row, *outputs, *expected;
    for(i=0; i<n_epoch; i++)
    {
        sum_error = 0;
        for(j=0; j < train_size; j++)
        {
            row = (float*) malloc(row_size * sizeof(float));
            assert(row != NULL);
            row = get_row(train, row, j, row_size);
            outputs = (float*) malloc((row_size-1) * sizeof(float));
            assert(outputs != NULL);
            outputs = forward_propagate(outputs, network, row, row_size-1);
            expected = (float*) malloc(n_outputs * sizeof(float));
            assert(expected != NULL);
            for(t=0; t<n_outputs; t++)
            {
                expected[t] = 0;
            }
            expected[(int)(row[row_size-1])] = 1;
            for(t=0; t < n_outputs; t++)
            {
                sum_error += pow((expected[t] - outputs[t]), 2);
            }
            free(outputs);
            backward_propagate_error(network, expected);
            update_weights(network, row, row_size, l_rate);
            free(row);
            free(expected);
        }
        printf("epoch = %d, lrate = %.3f, error = %.3f\n", i, l_rate, sum_error);
    }
    return network;
}

/*
* Function: predict
* -----------------------------
* Implement training of neural network.
*
* network - a given neural network.
* row - a row of the dataset.
* row_size - size of a row in the dataset.
*/
int predict(struct neural_network *network, float* row, int row_size, int n_outputs)
{
    int i, max_index;
    float *inputs;
    inputs = (float*) malloc(row_size * sizeof(float));
    inputs = forward_propagate(inputs, network, row, row_size);
    max_index = 0;
    for(i=0; i<n_outputs; i++)
    {
        if (inputs[i] > inputs[max_index])
        {
            max_index = i;
        }
    }
    free(inputs);
    return max_index;
}

/*
* Function: grade
* -----------------------------
* Calculate and return the accuracy percentage of the results.
*
* actual - the actual results.
* predicted - the predicted results.
* test_dataset_size - size of the test dataset.
*/
float grade(float *actual, int *predicted, int test_dataset_size)
{
    int i, correct;
    float final_grade;
    correct = 0;
    for(i=0; i<test_dataset_size; i++)
    {
        if (((int)(actual[i])) - predicted[i] == 0)
        {
            correct += 1;
        }
    }
    final_grade = ((float)(correct) / (float)(test_dataset_size)) * 100;
    return final_grade;
}

/*
* Function: get_row
* -----------------------------
* Given an array of rows, return the i'th row.
*
* dataset - array of arrays.
* row - will store the correct values here.
* i - the location of desired row.
* row_size - size of row.
*/
float* get_row(float **dataset, float *row, int i, int row_size)
{   
    int j;
    for(j=0; j < row_size; j++)
    {
        row[j] = dataset[i][j];
    }
    return row;
}

/*
* Function: copyArr1ToArr2
* -----------------------------
* Copy the n first elements of arr1 into n first places in arr2.
*
* arr1 - first array (the one which I will copy from).
* arr2 - second array (the one which I will copy to).
* n - length of arrays.
*/
void copyArr1ToArr2(float *arr1, float *arr2, int n)
{
    int i;
    for(i=0; i < n; i++)
    {
        arr2[i] = arr1[i];
    }
}

/*
* Function: print_network
* -----------------------------
* Print a given neural network.
*
* network - the neural network to be printed.
*/
void print_network(struct neural_network network)
{
    int i;
    struct layer* layers = network.layers;
    for(i=0; i < network.num_of_layers; i++)
    {
        print_layer(layers[i]);
        if (i != network.num_of_layers-1)
        {
            printf(";\n");
        }
    }
}

/*
* Function: print_layer
* -----------------------------
* Print a given layer of neurons.
*
* layer - the layer to be printed.
*/
void print_layer(struct layer layer)
{
    int i;
    struct neuron* neurons = layer.neurons;
    printf("Layer(");
    for(i=0; i < layer.num_of_neurons; i++)
    {
        print_neuron(neurons[i]);
        if (i != layer.num_of_neurons-1)
        {
            printf(", ");
        }
    }
    printf(")");
}

/*
* Function: print_neuron
* -----------------------------
* Print a given neuron.
*
* neuron - the neuron to be printed.
*/
void print_neuron(struct neuron neuron)
{
    float* weights;
    int i;
    printf("Neuron(Bias: %f, Output: %f, Delta: %f, Weights:", neuron.bias, neuron.output, neuron.delta);
    weights = neuron.weights;
    for(i=0; i < neuron.num_of_weights; i++)
    {
        printf("%f", weights[i]);
        if (i != neuron.num_of_weights - 1)
        {
            printf(", ");
        }
    }
    printf(")");
}

/*
* Function: free_network
* -----------------------------
* Free the space allocated for the given network.
*
* network - a pointer to the network to be freed from memory.
*/
void free_network(struct neural_network* network)
{
    int i,j;
    struct layer layer;
    struct neuron neuron;
    for(i=0; i < network->num_of_layers; i++)
    {
        layer = network->layers[i];
        for(j=0; j<layer.num_of_neurons; j++)
        {
            neuron = layer.neurons[j];
            free(neuron.weights);
        }
        free(layer.neurons);
    }
    free(network->layers);
}

/*
* Function: evaluate_network
* -----------------------------
* predict test set data and present the grade.
*
* network - a pointer to the network to be freed from memory.
* test_dataset - train data.
* test_actual - expected results for train data.
* test_dataset_size - number of rows in test dataset.
* row_size - size of each row in dataset.
* n_outputs - number of possible outputs.
*/
void evaluate_network(struct neural_network* network, float**test_dataset, float* test_actual, int test_dataset_size, int row_size, int n_outputs)
{
    int i, *predictions;
    float final_grade, *row;
    predictions = (int*) malloc(test_dataset_size * sizeof(int));
    for(i=0; i < test_dataset_size; i++)
    {
        row = (float*) malloc(row_size * sizeof(float));
        assert(row != NULL);
        row = get_row(test_dataset, row, i, row_size);
        predictions[i] = predict(network, row, row_size, n_outputs);
        free(row);
    }
    final_grade = grade(test_actual, predictions, test_dataset_size);
    free(predictions);
    printf("Grade of the Network for Test Dataset is: %f", final_grade);

}

/*
* Function: train_and_evaluate
* -----------------------------
* The function to be called from module. Initialize the model and train it using the provided dataset.
*
* training_dataset - the dataset that will be used for training the model.
* testing_dataset - the dataset that will be used for testing the trained model.
* test_actual - the expected results for the testing dataset.
* data_size - size of the test dataset.
* row_size - length of a single row from the training dataset.
* test_size - number of rows in the test datast.
* test_result_possibilities - number of different classifications that could be made.
*/
void train_and_evaluate(float **training_dataset, float **testing_dataset, float *test_actual, int data_size, int row_size, int test_size, int test_result_possibilities, int n_epoch, int n_hidden_layers, float l_rate)
{
    int i;
    struct neural_network network;
    for(i=0; i<test_size; i++){
        printf("test_actual[%i] is %f", i, test_actual[i]);
    }
    srand(42);
    network = initialize_network(network, row_size-1, 5, test_result_possibilities, n_hidden_layers);
    printf("Network Before Training:\n");
    print_network(network);
    printf("\n***********************************************************************************************************\n");
    network = *train_network(&network, training_dataset, data_size, row_size, l_rate, n_epoch, test_result_possibilities);
    printf("***********************************************************************************************************\n");
    printf("Network After Training:\n");
    print_network(network);
    printf("\n***********************************************************************************************************\n");
    evaluate_network(&network, testing_dataset, test_actual, test_size, row_size-1, test_result_possibilities);
    free_network(&network);
}