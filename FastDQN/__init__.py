try:
    from FastDQN.c_FastDQN import Agent
except ImportError:
    import sys
    print("Using slow python version, Please compile cython version for faster speed", file=sys.stderr)
    from FastDQN.py_FastDQN import Agent