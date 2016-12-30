from sys import version_info

if version_info[0] == 3:
    print("Python 3 detected")
    try:
        from FastDQN.c_FastDQN import Agent, test_methods
        print("Using Cython Version")
    except ImportError:
        print("Using slow python version, Please compile cython version for faster speed")
        from FastDQN.py_FastDQN import Agent
else:
    print("Python 2 detected")
    try:
        from c_FastDQN import Agent, test_methods
        print("Using Cython Version")
    except ImportError:
        print("Using slow python version, Please compile cython version for faster speed")
        from py_FastDQN import Agent