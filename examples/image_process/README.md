Image Process
=============

## Build and run

```shell script
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
$ mkdir build && cd build
$ cmake -DFOAM_USE_TBB=ON -DFOAM_USE_XSIMD=ON 
        -DXTENSOR_USE_TBB=ON -DXTENSOR_USE_XSIMD=ON --march=native 
        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} ../
$ make

$ export LD_LIBRARY_PATH=${CMAKE_PREFIX_PATH}/lib
$ ./image_process  # replace run with your executable file
```