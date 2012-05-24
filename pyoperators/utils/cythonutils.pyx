from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

__all__ = []

@cython.boundscheck(False)
def inspect_special_values_bool8(np.ndarray[np.uint8_t, ndim=1] v):
    cdef int nzeros = 0
    cdef unsigned int n = v.size
    cdef unsigned int i

    for i in range(n):
        if v[i] == 0:
            nzeros += 1
    return 0, nzeros, n - nzeros, False, nzeros in (0, n)

@cython.boundscheck(False)
def inspect_special_values_uint64(np.ndarray[np.uint64_t, ndim=1] v):
    cdef int nones = 0
    cdef int nzeros = 0
    cdef unsigned int n = v.size
    cdef unsigned int i
    cdef np.uint64_t value, value0 = v[0]
    cdef int same = 1
    cdef int other = 0

    for i in range(n):
        value = v[i]
        if value == 0:
            nzeros += 1
        elif value == 1:
            nones += 1
        else:
            other = 1
        if same == 1 and value != value0:
            same = 0
        if same == 0 and other == 1:
            return 0, 0, 0, True, False
    if other == 1:
        return 0, 0, 0, True, True
    return 0, nzeros, nones, False, same == 1

@cython.boundscheck(False)
def inspect_special_values_int64(np.ndarray[np.int64_t, ndim=1] v):
    cdef int nones = 0
    cdef int nzeros = 0
    cdef int nmones = 0
    cdef unsigned int n = v.size
    cdef unsigned int i
    cdef np.int64_t value, value0 = v[0]
    cdef int same = 1
    cdef int other = 0

    for i in range(n):
        value = v[i]
        if value == 0:
            nzeros += 1
        elif value == 1:
            nones += 1
        elif value == -1:
            nmones += 1
        else:
            other = 1
        if same == 1 and value != value0:
            same = 0
        if same == 0 and other == 1:
            return 0, 0, 0, True, False
    if other == 1:
        return 0, 0, 0, True, True
    return nmones, nzeros, nones, False, same == 1

@cython.boundscheck(False)
def inspect_special_values_float64(np.ndarray[np.float64_t, ndim=1] v):
    cdef int nones = 0
    cdef int nzeros = 0
    cdef int nmones = 0
    cdef unsigned int n = v.size
    cdef unsigned int i
    cdef np.float64_t value, value0 = v[0]
    cdef int same = 1
    cdef int other = 0

    for i in range(n):
        value = v[i]
        if value == 0:
            nzeros += 1
        elif value == 1:
            nones += 1
        elif value == -1:
            nmones += 1
        else:
            other = 1
        if same == 1 and value != value0:
            same = 0
        if same == 0 and other == 1:
            return 0, 0, 0, True, False
    if other == 1:
        return 0, 0, 0, True, True
    return nmones, nzeros, nones, False, same == 1

@cython.boundscheck(False)
def inspect_special_values_complex128(np.ndarray[np.complex128_t, ndim=1] v):
    cdef int nones = 0
    cdef int nzeros = 0
    cdef int nmones = 0
    cdef unsigned int n = v.size
    cdef unsigned int i
    cdef np.complex128_t value, value0 = v[0]
    cdef int same = 1
    cdef int other = 0

    for i in range(n):
        value = v[i]
        if value == 0:
            nzeros += 1
        elif value == 1:
            nones += 1
        elif value == -1:
            nmones += 1
        else:
            other = 1
        if same == 1 and value != value0:
            same = 0
        if same == 0 and other == 1:
            return 0, 0, 0, True, False
    if other == 1:
        return 0, 0, 0, True, True
    return nmones, nzeros, nones, False, same == 1

