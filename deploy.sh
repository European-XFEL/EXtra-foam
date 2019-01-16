#!/usr/bin/env bash

ENV_DIR=${PWD}/env
conda create -y -p ${ENV_DIR} python=3.6
${ENV_DIR}/bin/python -m pip install -e . -I

# In case karabo_data is not the latest version
git clone https://github.com/European-XFEL/karabo_data.git
cd karabo_data
${ENV_DIR}/bin/python -m pip install -e .
