#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "symnmf.h"

#define MEMORY_ERROR -1

static double **parse_matrix(PyObject *X, int rows, int cols)
{
    double **matrix = (double **)malloc(rows * sizeof(double *));
    if (!matrix)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix");
        return NULL;
    }

    for (int i = 0; i < rows; ++i)
    {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (!matrix[i])
        {
            for (int j = 0; j < i; ++j)
            {
                free(matrix[j]);
            }
            free(matrix);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix row");
            return NULL;
        }

        PyObject *row = PyList_GetItem(X, i);
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
            if (PyErr_Occurred())
            {
                for (int k = 0; k <= i; ++k)
                {
                    free(matrix[k]);
                }
                free(matrix);
                return NULL;
            }
        }
    }
    return matrix;
}

static PyObject *build_python_matrix(double **matrix, int rows, int cols)
{
    PyObject *py_matrix = PyList_New(rows);
    if (!py_matrix)
        return NULL;

    for (int i = 0; i < rows; ++i)
    {
        PyObject *row = PyList_New(cols);
        if (!row)
        {
            Py_DECREF(py_matrix);
            return NULL;
        }

        for (int j = 0; j < cols; ++j)
        {
            PyObject *val = PyFloat_FromDouble(matrix[i][j]);
            if (!val)
            {
                Py_DECREF(row);
                Py_DECREF(py_matrix);
                return NULL;
            }
            PyList_SET_ITEM(row, j, val);
        }
        PyList_SET_ITEM(py_matrix, i, row);
    }
    return py_matrix;
}

static PyObject *sym(PyObject *self, PyObject *args)
{
    int num, size;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiO", &num, &size, &X))
    {
        return NULL;
    }

    double **vectors = parse_matrix(X, num, size);
    if (!vectors)
        return NULL;

    double **sym_matrix = create_similarity_matrix(num, size, vectors);
    if (!sym_matrix)
    {
        free_matrix(vectors, num);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create similarity matrix");
        return NULL;
    }

    print_matrix(sym_matrix, num, num);
    free_matrix(vectors, num);
    free_matrix(sym_matrix, num);

    Py_RETURN_NONE;
}

static PyObject *ddg(PyObject *self, PyObject *args)
{
    int num, size;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiO", &num, &size, &X))
    {
        return NULL;
    }

    double **vectors = parse_matrix(X, num, size);
    if (!vectors)
        return NULL;

    double **ddg_matrix = create_diagonal_matrix(num, size, vectors);
    if (!ddg_matrix)
    {
        free_matrix(vectors, num);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create diagonal matrix");
        return NULL;
    }

    print_matrix(ddg_matrix, num, num);
    free_matrix(vectors, num);
    free_matrix(ddg_matrix, num);

    Py_RETURN_NONE;
}

static PyObject *norm(PyObject *self, PyObject *args)
{
    int num, size, need_to_print;
    PyObject *X;

    if (!PyArg_ParseTuple(args, "iiiO", &need_to_print, &num, &size, &X))
    {
        return NULL;
    }

    double **vectors = parse_matrix(X, num, size);
    if (!vectors)
        return NULL;

    double **norm_matrix = normalize_similarity_matrix(num, size, vectors);
    if (!norm_matrix)
    {
        free_matrix(vectors, num);
        PyErr_SetString(PyExc_RuntimeError, "Failed to normalize similarity matrix");
        return NULL;
    }

    PyObject *py_norm_matrix = NULL;
    if (need_to_print)
    {
        print_matrix(norm_matrix, num, num);
    }
    else
    {
        py_norm_matrix = build_python_matrix(norm_matrix, num, num);
    }

    free_matrix(vectors, num);
    free_matrix(norm_matrix, num);

    if (need_to_print)
    {
        Py_RETURN_NONE;
    }
    else
    {
        return py_norm_matrix;
    }
}

static PyObject *symnmf(PyObject *self, PyObject *args)
{
    int num, k, analysis;
    PyObject *H, *W;

    if (!PyArg_ParseTuple(args, "iiOOi", &k, &num, &W, &H, &analysis))
    {
        return NULL;
    }

    double **H_matrix = parse_matrix(H, num, k);
    if (!H_matrix)
        return NULL;

    double **norm_matrix = parse_matrix(W, num, num);
    if (!norm_matrix)
    {
        free_matrix(H_matrix, num);
        return NULL;
    }

    double **symnmf_matrix = calculate_symnmf(k, num, norm_matrix, H_matrix);
    if (!symnmf_matrix)
    {
        free_matrix(H_matrix, num);
        free_matrix(norm_matrix, num);
        PyErr_SetString(PyExc_RuntimeError, "Failed to calculate SYMNMF");
        return NULL;
    }

    PyObject *result = NULL;
    if (analysis)
    {
        result = build_python_matrix(symnmf_matrix, num, k);
    }
    else
    {
        print_matrix(symnmf_matrix, num, k);
        Py_INCREF(Py_None);
        result = Py_None;
    }

    free_matrix(H_matrix, num);
    free_matrix(norm_matrix, num);
    free_matrix(symnmf_matrix, num);

    return result;
}

static PyMethodDef symnmf_methods[] = {
    {"sym", (PyCFunction)sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", (PyCFunction)ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", (PyCFunction)norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", (PyCFunction)symnmf, METH_VARARGS, "Perform SYMNMF algorithm"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    "A Python module for SYMNMF algorithm",
    -1,
    symnmf_methods};

PyMODINIT_FUNC PyInit_mysymnmf(void)
{
    return PyModule_Create(&moduledef);
}
