from sys import version_info

if version_info[0] == 3:
    print("Python 3 detected")
    try:
        from FastDQN.c_FastDQN import Agent
        print("Using Cython Version")
    except:
        print("Please Compile the library by running compile.py first")
else:
    print("Python 2 detected")
    try:
        from c_FastDQN import Agent, test_methods
        print("Using Cython Version")
    except:
        print("Please Compile the library by running compile.py first")