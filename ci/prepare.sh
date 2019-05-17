#!/usr/bin/env bash

export http_proxy="http://exflwgs06.desy.de:3128/"
export https_proxy="http://exflwgs06.desy.de:3128/"

export DISPLAY=:99
start-stop-daemon --start --background --exec /usr/bin/Xvfb -- $DISPLAY -screen 0 1400x900x24

apt-get install -y libxi6 libxrender1 libxkbcommon-x11-0 libdbus-1-3

