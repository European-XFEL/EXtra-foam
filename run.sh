#!/bin/bash
docker run -it --rm -e DISPLAY=$DISPLAY -v ~/fxe-data:/fxe-data:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw --net="host" fxe-tools fxe-gui
