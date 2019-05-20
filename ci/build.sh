#!/usr/bin/env bash

conda create -n test_env python=3.7
source activate test_env
which python
python --version
pip install .[test]