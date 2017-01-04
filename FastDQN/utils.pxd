#!python
#cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True

######################
### Imports
######################
import numpy as np
cimport numpy as np
from libc.math cimport fabs


######################
### Typedefs
######################
from numpy cimport uint64_t as UINT64_t
from numpy cimport uint16_t as UINT16_t
from numpy cimport uint8_t as UINT8_t
from numpy cimport uint8_t as BOOL_t
from numpy cimport float32_t as FLOAT_t
from numpy cimport float64_t as DOUBLE_t


cdef FLOAT_t max_1d(FLOAT_t[::1] arr) nogil
cdef FLOAT_t sum_1d(FLOAT_t[::1] arr) nogil
cdef UINT64_t argmax_1d(FLOAT_t[::1] arr) nogil
cdef UINT16_t argmax_2d(FLOAT_t[:, ::1] arr) nogil

cdef void q_value(FLOAT_t[:, ::1] VS, FLOAT_t[:, ::1] VNS, UINT16_t[::1] A, FLOAT_t[::1] R,
                     UINT8_t[::1] T, FLOAT_t gamma, FLOAT_t[::1] Errors) nogil

cdef void double_q_value(FLOAT_t[:, ::1] VS, FLOAT_t[:, ::1] VSS, FLOAT_t[:, ::1] VNS, UINT16_t[::1] A, FLOAT_t[::1] R,
                         UINT8_t[::1] T, FLOAT_t gamma, FLOAT_t[::1] Errors) nogil