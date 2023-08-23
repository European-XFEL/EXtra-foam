Installation
============

If you want to use **EXtra-foam** on the online or `Maxwell` cluster, please check **GETTING STARTED**.

.. _Anaconda: https://www.anaconda.com/

To install **EXtra-foam** in your own environment, you are encouraged to use Anaconda_ to run
and build **EXtra-foam**.

.. _install-extra-foam:

Install **EXtra-foam**
----------------------

.. code-block:: bash

    $ git clone --recursive --branch <tag_name> https://github.com/European-XFEL/EXtra-foam.git

    # If you have cloned the repository without one or more of its submodules, run
    $ git submodule update --init

    $ cd EXtra-foam

    # Create an Anaconda_ environment, by default it's named 'extra_foam'. If
    # you want to install into an existing environment, use `conda env update` instead.
    $ conda env create -f environment.yml

    $ conda activate extra_foam
    # We need to set this variable so that the libraries from the conda
    # environment are loaded first, in particular libstdc++. If the system
    # libstdc++ is loaded first and it's too old for the version extra-foam was
    # compiled against, then we might get nasty loader errors.
    $ conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
    # Re-activate the environment so the new variables take effect
    $ conda deactivate
    $ conda activate extra_foam

    # Install extra-foam. If you don't have access to our Gitlab, you can also
    # install the default '.' target.
    $ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    $ pip install '.[correlator]'


Install your own **EXtra-foam** kernel on the Maxwell cluster for offline analysis
----------------------------------------------------------------------------------

For now, there is no documentation for the Python bindings of the C++ implementations in
**EXtra-foam**. If you are interested in using those super fast C++ implementation to
accelerate your analysis, please feel free to dig into the code and ask questions.

.. code-block:: bash

    # ssh to the Maxwell cluster and then
    $ module load anaconda3

    # follow the previous steps to install EXtra-foam

    $ conda install ipykernel nb_conda_kernels -y

    # Now you should be able to load the newly created kernel on max-jhub.


Install C++ API only
--------------------

.. _foamalgo: https://github.com/zhujun98/foamalgo

Please check foamalgo_.
