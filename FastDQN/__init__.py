from sys import version_info

if version_info[0] == 3:
    print("Python 3 detected")
    try:
        from FastDQN.FastDQN import MakeAgent
        from FastDQN.tests import tests
        print("Using Cython Version")
    except:
        print("Please Compile the library by running compile.py first")
else:
    print("Python 2 detected")
    try:
        from FastDQN import MakeAgent
        print("Using Cython Version")
    except:
        print("Please Compile the library by running compile.py first")