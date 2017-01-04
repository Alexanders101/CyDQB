#!/usr/bin/env bash
cd FastDQN
python3 compile.py build_ext --inplace
rm -r build/