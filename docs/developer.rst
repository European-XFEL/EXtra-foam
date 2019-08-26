DEVELOPER
=========

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



Release karaboFAI
"""""""""""""""""

- Update the **ChangeLog** in the `documentation` branch;
- Update the version number in `docs/conf.py` and `karaboFAI/__init__.py`;
- Merge the `documentation` branch into the `dev` branch;
- Merge the `dev` branch into the `master` branch;
- Tag the `master` branch.