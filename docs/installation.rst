INSTALLATION
============

If you are using karaboFAI on the online or `Maxwell` cluster, please check **GETTING STARTED**.

.. _Anaconda: https://www.anaconda.com/

To install **karaboFAI** in your own environment, you are encouraged to use Anaconda_ to run
and build **karaboFAI**.

Dependencies
------------

- Python >= 3.6
- cmake >= 3.8
- gcc >= 5.4 (support c++14)

In your Anaconda_ environment, run the following commands:

.. code-block:: bash

    $ conda install -c anaconda cmake numpy
    $ conda install -c omgarcia gcc-6


Install karaboFAI
-----------------

.. code-block:: bash

    $ git clone --recursive https://git.xfel.eu/gitlab/dataAnalysis/karaboFAI.git

    # If you have cloned the repository without one or more of its submodules, run
    $ git submodule update --init

    $ cd karaboFAI

    # optional
    $ export FAI_WITH_TBB=0  # turn off intel TBB in karaboFAI
    $ export XTENSOR_WITH_TBB=0  # turn off intel TBB in xtensor
    $ export FAI_WITH_XSIMD=0  # turn off XSIMD in karaboFAI
    $ export XTENSOR_WITH_XSIMD=0  # turn off XSIMD in xtensor

    # Note: This step is also required if one wants to change the above
    #       environmental parameters.
    $ python setup.py clean  # alternatively "rm -r build"

    $ pip install .
