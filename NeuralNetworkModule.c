#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "NeuralNetwork.c"

static PyObject* fit_capi(PyObject *self, PyObject *args)
{
    PyObject *train_data, *test_data, *test_actual, *prevDataPoint, *prevCord, *prevRes;
    float l_rate, **train, **test, *test_res, *DataPoint;
    int i, j, row_size, train_data_size, test_data_size, result_possibilities, n_epoch, n_hidden_layers;
    if(!PyArg_ParseTuple(args, "OOOiiiiiif", &train_data, &test_data, &test_actual, &train_data_size, &row_size, &test_data_size, &result_possibilities, &n_epoch, &n_hidden_layers, &l_rate))
    {
        Py_RETURN_NONE;
    }
    train = (float **) malloc(train_data_size * sizeof(float *));
    assert(train != NULL);
    test = (float **) malloc(test_data_size * sizeof(float *));
    assert(test != NULL);
    test_res = (float *) malloc(test_data_size * sizeof(float));
    assert(test_res != NULL);
    /*initialize train data:*/
    for (i = 0; i < train_data_size; i++)
    {
        prevDataPoint = PyList_GetItem(train_data, i);
        DataPoint = (float *) malloc(row_size * sizeof(float));
        assert(DataPoint != NULL);
        for (j=0; j < row_size; j++)
        {
            prevCord = PyList_GetItem(prevDataPoint, j);
            DataPoint[j] = (float)(PyFloat_AsDouble(prevCord));
        }
        train[i] = DataPoint;
    }
    /*initialize test data:*/
    for (i = 0; i < test_data_size; i++)
    {
        prevDataPoint = PyList_GetItem(test_data, i);
        DataPoint = (float *) malloc(row_size * sizeof(float));
        assert(DataPoint != NULL);
        for (j=0; j < row_size-1; j++)
        {
            prevCord = PyList_GetItem(prevDataPoint, j);
            DataPoint[j] = (float)(PyFloat_AsDouble(prevCord));
        }
        test[i] = DataPoint;
    }
    /*initialize test results:*/
    for (i = 0; i < test_data_size; i++)
    {
        prevRes = PyList_GetItem(test_actual, i);
        test_res[i] = (float)(PyFloat_AsDouble(prevRes));
    }
    train_and_evaluate(train, test, test_res, train_data_size, row_size, test_data_size, result_possibilities, n_epoch, n_hidden_layers, l_rate);
    Py_RETURN_NONE;
}

static PyMethodDef capiMethods[] = {
    {"fit", (PyCFunction) fit_capi, METH_VARARGS,
    PyDoc_STR("data points and sizes.")},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "NeuralNetworkModule",
    NULL,
    -1,
    capiMethods
};


PyMODINIT_FUNC
PyInit_NeuralNetworkModule(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m)
    {
        return NULL;
    }
    return m;
}
