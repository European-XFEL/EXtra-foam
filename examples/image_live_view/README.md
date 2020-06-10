Image Live View
===============

If you are not interested in the GUI code, you can simply check `Broker::recv` and
`ImageProcessor::process`.

## Dependencies

#### Install Qt5

```shell script
conda install -c conda-forge qt=5.12
```

#### Install [karabo-bridge-cpp](https://github.com/European-XFEL/karabo-bridge-cpp)

Follow the "Build and install" instruction.

## Build and run

```shell script
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
$ export LD_LIBRARY_PATH=${CMAKE_PREFIX_PATH}/lib

$ mkdir build && cd build
$ cmake -DFOAM_USE_TBB=ON -DFOAM_USE_XSIMD=ON 
        -DXTENSOR_USE_TBB=ON -DXTENSOR_USE_XSIMD=ON --march=native 
        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} ../
$ make
$ ./live_view
```
