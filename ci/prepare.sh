#!/usr/bin/env bash

set -x
set -e

export http_proxy="http://exflwgs06.desy.de:3128/"
export https_proxy="http://exflwgs06.desy.de:3128/"

export DISPLAY=:99
start-stop-daemon --start --background --exec /usr/bin/Xvfb -- $DISPLAY -screen 0 1400x900x24

apt update
# updated by Denivy in https://git.xfel.eu/gitlab/Karabo/ci-containers/merge_requests/15
#apt install -y build-essential cmake
#apt install -y libxi6 libxrender1 libxkbcommon-x11-0 libdbus-1-3

git submodule sync --recursive
git submodule update --init --recursive