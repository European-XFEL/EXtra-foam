#!/bin/bash


TP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"


if [[ -f ${TP_DIR}/build_redis.sh ]]; then
    bash ${TP_DIR}/build_redis.sh
fi
