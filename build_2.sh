#!/usr/bin/env bash
cd FastDQN
cython -a c_FastDQN.pyx
python compile.py build_ext --inplace
rm -r build/
rm *.c