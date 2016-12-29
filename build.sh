#!/usr/bin/env bash
cython -a c_FastDQN.pyx
python3 compile.py build_ext --inplace
rm -r build/
rm c_FastDQN.c