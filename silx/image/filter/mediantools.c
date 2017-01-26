/* Subset of SIGTOOLS module by Travis Oliphant

Copyright 2005 Travis Oliphant
Permission to use, copy, modify, and distribute this software without fee
is granted under the SciPy License.

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2009 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Enthought nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "Python.h"
/* adding next line may raise errors ...
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
*/
#include "numpy/noprefix.h"

#include <setjmp.h>

typedef struct {
  char *data;
  int elsize;
} Generic_ptr;

typedef struct {
  char *data;
  intp numels;
  int elsize;
  char *zero;        /* Pointer to Representation of zero */
} Generic_Vector;

typedef struct {
  char *data;
  int  nd;
  intp  *dimensions;
  int  elsize;
  intp  *strides;
  char *zero;         /* Pointer to Representation of zero */
} Generic_Array;

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif
#define PYERR(message)  \
        {struct module_state *st = GETSTATE(self);\
            PyErr_SetString(st->error, message);goto fail;}

jmp_buf MALLOC_FAIL;

char *check_malloc (int);

char *check_malloc (int size)
{
    char *the_block;

    the_block = (char *)malloc(size);
    if (the_block == NULL)
    {
        printf("\nERROR: unable to allocate %d bytes!\n", size);
        longjmp(MALLOC_FAIL,-1);
    }
    return(the_block);
}


static char doc_median2d[] = "filt = _median2d(data, size, conditional=0)";

extern void f_medfilt2(float*,float*,int*,int*,int);
extern void d_medfilt2(double*,double*,int*,int*,int);
extern void b_medfilt2(unsigned char*,unsigned char*,int*,int*,int);
extern void short_medfilt2(short*, short*,int*,int*,int);
extern void ushort_medfilt2(unsigned short*,unsigned short*,int*,int*,int);
extern void int_medfilt2(int*, int*,int*,int*,int);
extern void uint_medfilt2(unsigned int*,unsigned int*,int*,int*,int);
extern void long_medfilt2(long*, long*,int*,int*,int);
extern void ulong_medfilt2(unsigned long*,unsigned long*,int*,int*,int);

static PyObject *mediantools_median2d(PyObject *self, PyObject *args)
{
    PyObject *image=NULL, *size=NULL;
    int conditional_flag=0;
    int typenum;
    PyArrayObject *a_image=NULL, *a_size=NULL;
    PyArrayObject *a_out=NULL;
    int Nwin[2] = {3,3};
    long *lhelp;
    int Idims[2] = {0, 0};

    if (!PyArg_ParseTuple(args, "O|Oi", &image, &size, &conditional_flag)) return NULL;

    typenum = PyArray_ObjectType(image, 0);
    a_image = (PyArrayObject *)PyArray_ContiguousFromObject(image, typenum, 2, 2);
    if (a_image == NULL) goto fail;

    if (size != NULL) {
    a_size = (PyArrayObject *)PyArray_ContiguousFromObject(size, NPY_LONG, 1, 1);
    if (a_size == NULL) goto fail;
    if ((PyArray_NDIM(a_size) != 1) || (PyArray_DIMS(a_size)[0] < 2))
        PYERR("Size must be a length two sequence");
    lhelp = (long *) PyArray_DATA(a_size);
    Nwin[0] = (int) (*lhelp);
    Nwin[1] = (int) (*(lhelp++));
    Idims[0] = (int) (PyArray_DIMS(a_image)[0]);
    Idims[1] = (int) (PyArray_DIMS(a_image)[1]);
    }

    a_out = (PyArrayObject *)PyArray_SimpleNew(2,PyArray_DIMS(a_image),typenum);
    if (a_out == NULL) goto fail;

    if (setjmp(MALLOC_FAIL)) {
    PYERR("Memory allocation error.");
    }
    else {
    switch (typenum) {
    case NPY_UBYTE:
        b_medfilt2((unsigned char *)PyArray_DATA(a_image), (unsigned char *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    case NPY_FLOAT:
        f_medfilt2((float *)PyArray_DATA(a_image), (float *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    case NPY_DOUBLE:
        d_medfilt2((double *)PyArray_DATA(a_image), (double *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    case NPY_SHORT:
        short_medfilt2((short *)PyArray_DATA(a_image), (short *)PyArray_DATA(a_out),\
            Nwin, Idims, conditional_flag);
        break;
    case NPY_USHORT:
        ushort_medfilt2((unsigned short *)PyArray_DATA(a_image), (unsigned short *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    case NPY_INT:
        int_medfilt2((int *)PyArray_DATA(a_image), (int *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    case NPY_UINT:
        uint_medfilt2((unsigned int *)PyArray_DATA(a_image), (unsigned int *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    case NPY_LONG:
        long_medfilt2((long *)PyArray_DATA(a_image), (long *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    case NPY_ULONG:
        ulong_medfilt2((unsigned long *)PyArray_DATA(a_image), (unsigned long *)PyArray_DATA(a_out),\
                   Nwin, Idims, conditional_flag);
        break;
    default:
      PYERR("Median filter unsupported data type.");
    }
    }

    Py_DECREF(a_image);
    Py_XDECREF(a_size);

    return PyArray_Return(a_out);

 fail:
    Py_XDECREF(a_image);
    Py_XDECREF(a_size);
    Py_XDECREF(a_out);
    return NULL;

}

static struct PyMethodDef mediantools_methods[] = {
    {"_medfilt2d", mediantools_median2d, METH_VARARGS, doc_median2d},
    {NULL,        NULL, 0}        /* sentinel */
};

/* Initialization function for the module (*must* be called initmediantools) */
/* Module initialization */

#if PY_MAJOR_VERSION >= 3

static int mediantools_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int mediantools_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mediantools",
        NULL,
        sizeof(struct module_state),
        mediantools_methods,
        NULL,
        mediantools_traverse,
        mediantools_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_mediantools(void)

#else
#define INITERROR return

void
initmediantools(void)
#endif
{
    struct module_state *st;
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("mediantools", mediantools_methods);
#endif

    if (module == NULL)
        INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("mediantools.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    import_array();
    PyImport_ImportModule("numpy.core.multiarray");
    /* Check for errors */
    if (PyErr_Occurred()) {
      PyErr_Print();
      Py_FatalError("can't initialize module array");
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
