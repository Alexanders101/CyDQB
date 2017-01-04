#!/usr/bin/env bash
cd FastDQN
python compile.py build_ext --inplace
rm -r build/
rm *.c