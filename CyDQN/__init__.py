from sys import version_info

if version_info[0] == 3:
    print("Python 3 detected")
    try:
        from CyDQN.FastDQN import MakeAgent
        from CyDQN.tests import tests
        print("Using Cython Version")
    except:
        print("Please Compile the library by running compile.py first")
else:
    print("Python 2 detected")
    try:
        from CyDQN import MakeAgent
        print("Using Cython Version")
    except:
        print("Please Compile the library by running compile.py first")