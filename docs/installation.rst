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
    # $ conda install -c conda-forge compilers

    # install cmake and dependencies
    $ conda install -c anaconda cmake numpy
    $ conda install -c conda-forge "tbb-devel<2021.1"

Install **EXtra-foam**
----------------------

.. code-block:: bash

    $ git clone --recursive --branch <tag_name> https://github.com/European-XFEL/EXtra-foam.git

    # If you have cloned the repository without one or more of its submodules, run
    $ git submodule update --init

    $ cd EXtra-foam
    $ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    $ pip install .


Install your own **EXtra-foam** kernel on the Maxwell cluster for offline analysis
----------------------------------------------------------------------------------

For now, there is no documentation for the Python bindings of the C++ implementations in
**EXtra-foam**. If you are interested in using those super fast C++ implementation to
accelerate your analysis, please feel free to dig into the code and ask questions.

.. code-block:: bash

    # ssh to the Maxwell cluster and then
    $ module load anaconda3
    $ conda create --name extra-foam-offline -y
    $ conda activate extra-foam-offline

    # follow the previous steps to install EXtra-foam

    $ conda install ipykernel nb_conda_kernels -y

    # Now you should be able to load the newly created kernel on max-jhub.


Install C++ API only
--------------------

.. _foamalgo: https://github.com/zhujun98/foamalgo

Please check foamalgo_.
