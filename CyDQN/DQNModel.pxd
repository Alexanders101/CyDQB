#!python
#cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True

######################
### Imports
######################
from CyDQN.utils cimport *

cdef class DQNModel:
    cdef FLOAT_t gamma
    cdef BOOL_t double_dqn
    cdef object model
    cdef object target_model

    cdef FLOAT_t[:, ::1] predict(self, UINT8_t[:,:,:,::1] x)
    cdef FLOAT_t fit(self, UINT8_t[:,:,:,::1] S, UINT8_t[:,:,:,::1] NS, UINT16_t[::1] A, FLOAT_t[::1] R,
                          UINT8_t[::1] T, FLOAT_t[::1] Errors, FLOAT_t[::1] weights)
    cdef str summary(self)
    cdef void hard_update_target_weights(self)
    cdef void save_weights(self, str filepath, BOOL_t overwrite=?)
    cdef void load_weights(self, str filepath)
