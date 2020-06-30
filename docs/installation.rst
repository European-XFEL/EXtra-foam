INSTALLATION
============

If you want to use **EXtra-foam** on the online or `Maxwell` cluster, please check **GETTING STARTED**.

.. _Anaconda: https://www.anaconda.com/

To install **EXtra-foam** in your own environment, you are encouraged to use Anaconda_ to run
and build **EXtra-foam**.

Dependencies
------------

- Python >= 3.6
- cmake >= 3.8
- gcc >= 5.4 (support c++14)

In your Anaconda_ environment, run the following commands:

.. code-block:: bash

    # install gcc-6.1
    $ conda install -c omgarcia gcc-6
    $ conda install -c anaconda libstdcxx-ng

    # install gcc-7.5 (experimenting)
    # $ conda install -c conda-forge gxx_linux-64
    # $ ln -s ${CONDA_PREFIX}/bin/x86_64-conda_cos6-linux-gnu-gcc  ${CONDA_PREFIX}/bin/gcc
    # $ ln -s ${CONDA_PREFIX}/bin/x86_64-conda_cos6-linux-gnu-g++  ${CONDA_PREFIX}/bin/g++

    # install cmake and dependencies
    $ conda install -c anaconda cmake numpy
    $ conda install -c conda-forge tbb-devel

Install **EXtra-foam**
----------------------

.. code-block:: bash

    $ git clone --recursive --branch <tag_name> https://github.com/European-XFEL/EXtra-foam.git

    # If you have cloned the repository without one or more of its submodules, run
    $ git submodule update --init

    $ cd EXtra-foam

    # optional
    $ export FOAM_USE_TBB=0  # turn off intel TBB in extra-foam
    $ export XTENSOR_USE_TBB=0  # turn off intel TBB in xtensor
    $ export FOAM_USE_XSIMD=0  # turn off XSIMD in extra-foam
    $ export XTENSOR_USE_XSIMD=0  # turn off XSIMD in xtensor

    # Note: This step is also required if one wants to change the above
    #       environmental parameters.
    $ python setup.py clean  # alternatively "rm -r build"

    $ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    $ pip install .


Install C++ API of **EXtra-foam** only
--------------------------------------

.. code-block:: bash

    $ mkdir build && cd build
    $ cmake -DFOAM_USE_XSIMD=ON -DCMAKE_INSTALL_PREFIX=/YOUR/INSTALL/PREFIX
    $ make && make install
