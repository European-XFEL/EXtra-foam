#!/bin/bash

set -x
set -e


TP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../
ROOT_DIR=${TP_DIR}/..

REDIS_VERSION=5.0.4
REDIS_BUILD_DIR=${TP_DIR}/pkg/redis/

if [[ ! -f ${REDIS_BUILD_DIR}/src/redis-server ]]; then

    REDIS_DOWNLOAD=${REDIS_VERSION}.tar.gz
    wget https://github.com/antirez/redis/archive/${REDIS_DOWNLOAD} -P ${TP_DIR}

    mkdir -p ${REDIS_BUILD_DIR}
    tar -xzf ${TP_DIR}/${REDIS_DOWNLOAD} --strip-components=1 -C ${REDIS_BUILD_DIR}
    rm ${TP_DIR}/${REDIS_DOWNLOAD}

    pushd ${REDIS_BUILD_DIR}
        make
    popd
fi

mkdir -p ${ROOT_DIR}/extra_foam/thirdparty/bin
cp ${REDIS_BUILD_DIR}/src/redis-server ${ROOT_DIR}/extra_foam/thirdparty/bin/redis-server
cp ${REDIS_BUILD_DIR}/src/redis-cli ${ROOT_DIR}/extra_foam/thirdparty/bin/redis-cli
