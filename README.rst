FXE tools for train integration
===============================

This tool has only been tested under Python3.6 environment.

Installation
------------

.. code-block:: bash

    $ pip install -e .


GUI
---

To start the GUI, simply type:

.. code-block:: bash

    $ fxe-gui


Start a file server
-------------------

One can start a virtual bridge by streaming from files:

.. code-block:: bash

    $ karabo-bridge-serve-files dirname port

In the meanwhile, one should select ``Calibrated (file)`` as the
``Data source`` in the GUI.
