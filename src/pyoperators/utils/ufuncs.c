/*
 * NumPy ufuncs for pyoperators
 * Converted from ufuncs.c.src to use C preprocessor macros instead of numpy.distutils templates
 * This enables compatibility with NumPy 2.0+ and modern build systems like Meson
 */

#include <Python.h>
#include <math.h>
#include "numpy/npy_math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#define UNARY_LOOP\
    char *ip = args[0], *op = args[1];\
    npy_intp is = steps[0], os = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip += is, op += os)

#define BINARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)

static char complex1_float1_types[12] = {NPY_CFLOAT, NPY_FLOAT,
                                         NPY_CDOUBLE, NPY_DOUBLE,
                                         NPY_CLONGDOUBLE, NPY_LONGDOUBLE,
                                         NPY_FLOAT, NPY_FLOAT,
                                         NPY_DOUBLE, NPY_DOUBLE,
                                         NPY_LONGDOUBLE, NPY_LONGDOUBLE};
static char float2_types[9] = {NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
                               NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                               NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE};
static char complex2_types[18] =
    {NPY_CFLOAT, NPY_CFLOAT, NPY_CFLOAT,
     NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
     NPY_CLONGDOUBLE, NPY_CLONGDOUBLE, NPY_CLONGDOUBLE,
     NPY_FLOAT, NPY_CFLOAT, NPY_CFLOAT,
     NPY_DOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
     NPY_LONGDOUBLE, NPY_CLONGDOUBLE, NPY_CLONGDOUBLE};

static void *null_data3[3] = {NULL, NULL, NULL};
static void *null_data6[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
static void *null_data17[17] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                                NULL};


/*********************
 * Complex abs(x)**2 *
 *********************/

/* Macro to generate abs2 functions for complex types */
#define DEFINE_ABS2(ftype, suffix) \
NPY_NO_EXPORT void \
abs2##suffix(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) \
{ \
    UNARY_LOOP { \
        const ftype inr = *(ftype *)ip; \
        const ftype ini = ((ftype *)ip)[1]; \
        *((ftype *)op) = inr*inr + ini*ini; \
    } \
} \
 \
NPY_NO_EXPORT void \
abs2##suffix##_real(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) \
{ \
    UNARY_LOOP { \
        const ftype in = *(ftype *)ip; \
        *((ftype *)op) = in * in; \
    } \
}

DEFINE_ABS2(npy_float, f)
DEFINE_ABS2(npy_double, )
DEFINE_ABS2(npy_longdouble, l)

static PyUFuncGenericFunction abs2_funcs[6] =
    {&abs2f, &abs2, &abs2l,
     &abs2f_real, &abs2_real, &abs2l_real};


/*********************
 * Hard thresholding *
 *********************/

/* Macro to generate hard_thresholding functions */
#define DEFINE_HARD_THRESHOLDING(type, suffix) \
NPY_NO_EXPORT void \
hard_thresholding##suffix(char **args, const npy_intp *dimensions, const npy_intp* steps, \
                     void* data) \
{ \
    npy_intp i; \
    npy_intp n = dimensions[0]; \
    char *in = args[0], *threshold = args[1], *out = args[2]; \
    npy_intp in_step = steps[0], threshold_step = steps[1], out_step = steps[2]; \
 \
    type tmp; \
 \
    for (i = 0; i < n; i++) { \
        tmp = *(type *)in; \
        tmp = (fabs##suffix(tmp) > *(type *)threshold) ? tmp : 0; \
        *((type *)out) = tmp; \
 \
        in += in_step; \
        threshold += threshold_step; \
        out += out_step; \
    } \
}

DEFINE_HARD_THRESHOLDING(npy_float, f)
DEFINE_HARD_THRESHOLDING(npy_double, )
DEFINE_HARD_THRESHOLDING(npy_longdouble, l)

static PyUFuncGenericFunction hard_thresholding_funcs[3] =
           {&hard_thresholdingf,
            &hard_thresholding,
            &hard_thresholdingl};


/***********
 * Masking *
 ***********/

/* Macro to generate masking functions for real types */
#define DEFINE_MASKING(TYPE, type) \
NPY_NO_EXPORT void \
TYPE##_masking(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data) \
{ \
    npy_intp i; \
    npy_intp n = dimensions[0]; \
    char *in = args[0], *mask = args[1], *out = args[2]; \
    npy_intp in_step = steps[0], mask_step = steps[1], out_step = steps[2]; \
 \
    if (in == out) { \
        for (i = 0; i < n; i++) { \
            if (*mask) \
                *((type *)out) = 0; \
            mask += mask_step; \
            out += out_step; \
        } \
    } else { \
        for (i = 0; i < n; i++) { \
            if (*mask) \
                *((type *)out) = 0; \
            else \
                *((type *)out) = *(type *)in; \
            in += in_step; \
            mask += mask_step; \
            out += out_step; \
        } \
    } \
}

DEFINE_MASKING(BYTE, npy_byte)
DEFINE_MASKING(UBYTE, npy_ubyte)
DEFINE_MASKING(SHORT, npy_short)
DEFINE_MASKING(USHORT, npy_ushort)
DEFINE_MASKING(INT, npy_int)
DEFINE_MASKING(UINT, npy_uint)
DEFINE_MASKING(LONG, npy_long)
DEFINE_MASKING(ULONG, npy_ulong)
DEFINE_MASKING(LONGLONG, npy_longlong)
DEFINE_MASKING(ULONGLONG, npy_ulonglong)
DEFINE_MASKING(HALF, npy_half)
DEFINE_MASKING(FLOAT, npy_float)
DEFINE_MASKING(DOUBLE, npy_double)
DEFINE_MASKING(LONGDOUBLE, npy_longdouble)

/* Macro to generate masking functions for complex types */
#define DEFINE_MASKING_COMPLEX(TYPE, type, ftype) \
NPY_NO_EXPORT void \
TYPE##_masking(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data) \
{ \
    npy_intp i; \
    npy_intp n = dimensions[0]; \
    char *in = args[0], *mask = args[1], *out = args[2]; \
    npy_intp in_step = steps[0], mask_step = steps[1], out_step = steps[2]; \
 \
    if (in == out) { \
        for (i = 0; i < n; i++) { \
            if (*mask) { \
                ((ftype *)out)[0] = 0.; \
                ((ftype *)out)[1] = 0.; \
            } \
            mask += mask_step; \
            out += out_step; \
        } \
    } else { \
        for (i = 0; i < n; i++) { \
            if (*mask) { \
                ((ftype *)out)[0] = 0.; \
                ((ftype *)out)[1] = 0.; \
            } else \
                *((type *)out) = *(type *)in; \
            in += in_step; \
            mask += mask_step; \
            out += out_step; \
        } \
    } \
}

DEFINE_MASKING_COMPLEX(CFLOAT, npy_cfloat, npy_float)
DEFINE_MASKING_COMPLEX(CDOUBLE, npy_cdouble, npy_double)
DEFINE_MASKING_COMPLEX(CLONGDOUBLE, npy_clongdouble, npy_longdouble)

static PyUFuncGenericFunction masking_funcs[17] =
           {&BYTE_masking, &UBYTE_masking,
            &SHORT_masking, &USHORT_masking,
            &INT_masking, &UINT_masking,
            &LONG_masking, &ULONG_masking,
            &LONGLONG_masking, &ULONGLONG_masking,
            &HALF_masking, &FLOAT_masking,
            &DOUBLE_masking, &LONGDOUBLE_masking,
            &CFLOAT_masking, &CDOUBLE_masking,
            &CLONGDOUBLE_masking};

static char masking_types[17*3] = {NPY_BYTE, NPY_BOOL, NPY_BYTE,
                                   NPY_UBYTE, NPY_BOOL, NPY_UBYTE,
                                   NPY_SHORT, NPY_BOOL, NPY_SHORT,
                                   NPY_USHORT, NPY_BOOL, NPY_USHORT,
                                   NPY_INT, NPY_BOOL, NPY_INT,
                                   NPY_UINT, NPY_BOOL, NPY_UINT,
                                   NPY_LONG, NPY_BOOL, NPY_LONG,
                                   NPY_ULONG, NPY_BOOL, NPY_ULONG,
                                   NPY_LONGLONG, NPY_BOOL, NPY_LONGLONG,
                                   NPY_ULONGLONG, NPY_BOOL, NPY_ULONGLONG,
                                   NPY_HALF, NPY_BOOL, NPY_HALF,
                                   NPY_FLOAT, NPY_BOOL, NPY_FLOAT,
                                   NPY_DOUBLE, NPY_BOOL, NPY_DOUBLE,
                                   NPY_LONGDOUBLE, NPY_BOOL, NPY_LONGDOUBLE,
                                   NPY_CFLOAT, NPY_BOOL, NPY_CFLOAT,
                                   NPY_CDOUBLE, NPY_BOOL, NPY_CDOUBLE,
                                   NPY_CLONGDOUBLE, NPY_BOOL, NPY_CLONGDOUBLE};


/****************************
 * Conjugate multiplication *
 ****************************/

/* Macro to generate multiply_conjugate functions */
#define DEFINE_MULTIPLY_CONJUGATE(ftype, suffix) \
NPY_NO_EXPORT void \
multiply_conjugate##suffix(char **args, const npy_intp *dimensions, const npy_intp *steps, \
                      void *data) \
{ \
    BINARY_LOOP { \
        const ftype in1r = *(ftype *)ip1; \
        const ftype in1i = ((ftype *)ip1)[1]; \
        const ftype in2r = ((ftype *)ip2)[0]; \
        const ftype in2i = ((ftype *)ip2)[1]; \
        ((ftype *)op1)[0] =  in1r*in2r + in1i*in2i; \
        ((ftype *)op1)[1] = -in1r*in2i + in1i*in2r; \
    } \
} \
 \
NPY_NO_EXPORT void \
multiply_real_conjugate##suffix(char **args, const npy_intp *dimensions, const npy_intp *steps, \
                           void *data) \
{ \
    BINARY_LOOP { \
        const ftype in1r = ((ftype *)ip1)[0]; \
        const ftype in2r = ((ftype *)ip2)[0]; \
        const ftype in2i = ((ftype *)ip2)[1]; \
        ((ftype *)op1)[0] =  in1r*in2r; \
        ((ftype *)op1)[1] = -in1r*in2i; \
    } \
}

DEFINE_MULTIPLY_CONJUGATE(npy_float, f)
DEFINE_MULTIPLY_CONJUGATE(npy_double, )
DEFINE_MULTIPLY_CONJUGATE(npy_longdouble, l)

static PyUFuncGenericFunction multiply_conjugate_funcs[6] =
           {&multiply_conjugatef,
            &multiply_conjugate,
            &multiply_conjugatel,
            &multiply_real_conjugatef,
            &multiply_real_conjugate,
            &multiply_real_conjugatel};


/*********************
 * Soft thresholding *
 *********************/

/* Macro to generate soft_thresholding functions */
#define DEFINE_SOFT_THRESHOLDING(type, suffix) \
NPY_NO_EXPORT void \
soft_thresholding##suffix(char **args, const npy_intp *dimensions, const npy_intp* steps, \
                     void* data) \
{ \
    npy_intp i; \
    npy_intp n = dimensions[0]; \
    char *in = args[0], *threshold = args[1], *out = args[2]; \
    npy_intp in_step = steps[0], threshold_step = steps[1], out_step = steps[2]; \
 \
    type tmp; \
 \
    for (i = 0; i < n; i++) { \
        tmp = fabs##suffix(*(type *)in) - *(type *)threshold; \
        tmp = (tmp > 0) ? tmp : 0; \
        *((type *)out) = copysign##suffix(tmp, *(type *)in); \
 \
        in += in_step; \
        threshold += threshold_step; \
        out += out_step; \
    } \
}

DEFINE_SOFT_THRESHOLDING(npy_float, f)
DEFINE_SOFT_THRESHOLDING(npy_double, )
DEFINE_SOFT_THRESHOLDING(npy_longdouble, l)

static PyUFuncGenericFunction soft_thresholding_funcs[3] =
           {&soft_thresholdingf,
            &soft_thresholding,
            &soft_thresholdingl};


/* Module definition */

static PyMethodDef module_methods[] = {
    { NULL, NULL, 0, NULL }
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufuncs",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_ufuncs()

{
    PyObject *m, *f, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    d = PyModule_GetDict(m);

    f = PyUFunc_FromFuncAndData(abs2_funcs, null_data6,
            complex1_float1_types, 6, 1, 1, PyUFunc_None, "abs2",
            "Compute y = x.real**2 + x.imag**2", 0);
    PyDict_SetItemString(d, "abs2", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(hard_thresholding_funcs, null_data3,
            float2_types, 3, 2, 1, PyUFunc_None, "hard_thresholding",
            "Compute y = x1 if |x1| > x2,\n            0  otherwise." , 0);
    PyDict_SetItemString(d , "hard_thresholding", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(masking_funcs, null_data17,
            masking_types, 17, 2, 1, PyUFunc_None, "masking",
            "Set y = 0 where x2,\n        x1  otherwise." , 0);
    PyDict_SetItemString(d , "masking", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(multiply_conjugate_funcs, null_data3,
            complex2_types, 3, 2, 1, PyUFunc_None, "multiply_conjugate",
            "Compute y = x1 * conjugate(x2)", 0);
    PyDict_SetItemString(d, "multiply_conjugate", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(soft_thresholding_funcs, null_data3,
            float2_types, 3, 2, 1, PyUFunc_None, "soft_thresholding",
            "Compute y = sign(x1) * [|x1| - x2]+" , 0);
    PyDict_SetItemString(d , "soft_thresholding", f);
    Py_DECREF(f);

    return m;
}
