#!/usr/bin/env bash

set -x
set -e

conda create -n test_env python=3.7
source activate test_env
which python
python --version

# although gcc-5 should be enought, but it is safer to go to gcc-6
# to have full support of c++14
conda install -c omgarcia gcc-6
g++ --version

# we need cmake >=3.8
conda install -c anaconda cmake
cmake --version

# we need "-e" here to make the test work for modules from C++
# better solutions?
which pip
pip --version
pip install -e .[test]