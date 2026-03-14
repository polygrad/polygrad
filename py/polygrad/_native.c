/*
 * _native.c -- Minimal Python extension stub.
 *
 * All polygrad C sources are compiled into this extension module.
 * The actual API is accessed via ctypes in _ffi.py.
 * This file just provides the PyInit entry point so Python
 * can load the .so as a proper extension module.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "polygrad native C core",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&native_module);
}
