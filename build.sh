#!/bin/bash


set -x
set -e

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

BUILD_DIR=${ROOT_DIR}/build
if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir ${BUILD_DIR}
fi

bash $ROOT_DIR/thirdparty/scripts/build_all.sh

pushd ${BUILD_DIR}

popd