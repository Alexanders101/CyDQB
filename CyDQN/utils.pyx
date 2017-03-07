#!python
#cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True
import numpy as np

cdef FLOAT_t max_1d(FLOAT_t[::1] arr) nogil:
    """ Calculate the maximum element in a 1-dim float array
    Parameters
    ----------
    arr: Numpy array
    size: size of array

    Returns
    -------
    Maximum Value
    """
    cdef UINT64_t i
    cdef FLOAT_t max = arr[0]
    cdef FLOAT_t cur_val

    for i in range(1, arr.shape[0]):
        cur_val = arr[i]
        if cur_val > max:
            max = cur_val

    return max

cdef FLOAT_t sum_1d(FLOAT_t[::1] arr) nogil:
    """ Calculate the maximum element in a 1-dim float array
    Parameters
    ----------
    arr: Numpy array
    size: size of array

    Returns
    -------
    Maximum Value
    """
    cdef UINT64_t i
    cdef FLOAT_t sum = 0

    for i in range(0, arr.shape[0]):
        sum += arr[i]

    return sum

cdef UINT64_t argmax_1d(FLOAT_t[::1] arr) nogil:
    cdef FLOAT_t max = arr[0]
    cdef UINT64_t max_index = 0

    cdef UINT64_t i
    cdef FLOAT_t cur_val

    for i in range(1, arr.shape[0]):
        cur_val = arr[i]
        if cur_val > max:
            max = cur_val
            max_index = i

    return max_index


cdef UINT16_t argmax_2d(FLOAT_t[:, ::1] arr) nogil:
    cdef FLOAT_t max = arr[0, 0]
    cdef UINT16_t max_index = 0

    cdef UINT16_t i
    cdef FLOAT_t cur_val

    for i in range(1, arr.shape[1]):
        cur_val = arr[0, i]
        if cur_val > max:
            max = cur_val
            max_index = i

    return max_index


cdef void double_q_value(FLOAT_t[:, ::1] VS, FLOAT_t[:, ::1] VSS, FLOAT_t[:, ::1] VNS, UINT16_t[::1] A, FLOAT_t[::1] R,
                         UINT8_t[::1] T, FLOAT_t gamma, FLOAT_t[::1] Errors) nogil:
    cdef UINT64_t batch_size = VS.shape[0]
    cdef UINT64_t action_count = VS.shape[1]
    cdef FLOAT_t value

    cdef UINT64_t i
    for i in range(batch_size):
        value = R[i] + (gamma * (1 - T[i]) * VNS[i, argmax_1d(VSS[i])])
        Errors[i] = fabs(VS[i, A[i]] - value)
        VS[i, A[i]] = value

cdef void q_value(FLOAT_t[:, ::1] VS, FLOAT_t[:, ::1] VNS, UINT16_t[::1] A, FLOAT_t[::1] R,
                     UINT8_t[::1] T, FLOAT_t gamma, FLOAT_t[::1] Errors) nogil:
    cdef UINT64_t batch_size = VS.shape[0]
    cdef UINT64_t action_count = VS.shape[1]
    cdef FLOAT_t value

    cdef UINT64_t i
    for i in range(batch_size):
        value = R[i] + (gamma * (1 - T[i]) * max_1d(VNS[i]))
        Errors[i] = fabs(VS[i, A[i]] - value)
        VS[i, A[i]] = value

