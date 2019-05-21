#!/usr/bin/env bash

set -x
set -e

conda create -n test_env python=3.7
source activate test_env
which python
python --version
# we need "-e" here to make the test work for modules from C++
# better solutions?
pip install -e .[test]