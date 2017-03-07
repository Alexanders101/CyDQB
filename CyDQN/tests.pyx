import numpy as np
cimport numpy as np
from FastDQN.utils cimport *

class tests:

    @staticmethod
    def numpy_copy_test():
        from time import time
        cdef np.uint8_t[::1] a = np.random.randint(0,250,1000000, np.uint8)
        cdef np.ndarray b

        t0 = time()
        b = np.asarray(a)
        t1 = time()
        print(t1-t0)

    @staticmethod
    def utils_test():
        a = np.array([5,7,1,1,10,2], dtype=np.float32)
        assert max_1d(a) == 10
        assert argmax_1d(a) == 4
        assert sum_1d(a) == 26

        a = a.reshape(1, 6)
        assert argmax_2d(a) == 4

