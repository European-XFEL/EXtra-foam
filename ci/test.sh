#!/usr/bin/env bash

set -x
set -e

# test parallel version
python setup.py build_ext --with-tests
python setup.py test

# test series version
export FOAM_WITH_TBB=0
export FOAM_WITH_XSIMD=0
export XTENSOR_WITH_TBB=0
export XTENSOR_WITH_XSIMD=0
python setup.py build_ext --with-tests
python setup.py test
