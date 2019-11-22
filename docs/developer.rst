DEVELOPER
=========

Design
""""""

.. image:: images/design_overview_developer.png
   :width: 800


Build and Test
""""""""""""""

.. _GoogleTest: https://github.com/google/googletest

Before running the Python unittest, rebuild the c++ code if it was updated. One can do either

.. code-block:: bash

    $ pip install -e . -v

or

.. code-block:: bash

    $ python setup.py build_ext --inplace

Then, run the Python unittest:

.. code-block:: bash

    $ python -m pytest karaboFAI -v -s


To build and run the c++ unittest (we use GoogleTest_):

.. code-block:: bash

    $ mkdir build && cd build
    $ cmake -DBUILD_FAI_TESTS .. && make ftest



Release **karaboFAI**
"""""""""""""""""""""

- Update the **ChangeLog** in the `documentation` branch;
- Update the version number in `docs/conf.py` and `karaboFAI/__init__.py`;
- Merge the `documentation` branch into the `dev` branch;
- Merge the `dev` branch into the `master` branch;
- Tag the `master` branch.


Deployment on Exfel Anaconda Environment
""""""""""""""""""""""""""""""""""""""""

**karaboFAI** deployment on exfel anaconda environments should be done using
**xsoft** account. Use the following anaconda environments to deploy particular
versions of **karaboFAI**

.. list-table::
   :header-rows: 1

   * - Version
     - Deployment environment

   * - Latest
     - exfel_anaconda3/beta.

   * - Stable
     - karaboFAI

   * - some-feature-branch (only for developers to test new features)
     - karaboFAI/beta


.. code-block:: console

   $ ssh xsoft@max-display.desy.de
   $ cd workspace
   $ git clone --recursive --branch <tag_name> ssh://git@git.xfel.eu:10022/dataAnalysis/karaboFAI.git karaboFAI-<tag_name>
   $ cd karaboFAI-<tag_name>
   $ module load exfel <environment_name>
   $ which pip
   /gpfs/exfel/sw/software/<environment_name>/bin/pip
   $ pip install . -v

.. note::

   ssh to the Maxwell and online cluster with your own account, 
   respectively, and launch **karaboFAI** there to double check the deployed version.
