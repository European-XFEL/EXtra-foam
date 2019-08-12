#!/usr/bin/env bash

set -x
set -e

python setup.py build_ext --with-tests
python setup.py test