#!python
#cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True

from FastDQN.utils cimport *
from FastDQN.DQNModel cimport DQNModel

cdef class Agent:
    cdef object game
    cdef DQNModel model

    cdef tuple frame_size
    cdef tuple state_size
    cdef UINT64_t num_actions
    cdef UINT64_t channel_axis
    cdef UINT64_t colors

    cdef UINT64_t frame_seq_count
    cdef UINT64_t save_freq
    cdef UINT64_t memory
    cdef UINT64_t batch_size
    cdef UINT64_t frame_skip
    cdef DOUBLE_t epsilon
    cdef DOUBLE_t delta_epsilon
    cdef FLOAT_t gamma
    cdef FLOAT_t tau
    cdef FLOAT_t alpha
    cdef FLOAT_t beta

    cdef str save_name

    cdef public list scores
    cdef public list losses

    cdef void play_(self, UINT64_t num_episodes, BOOL_t train=?)
    cpdef play(self, UINT64_t num_episodes, BOOL_t train=?)