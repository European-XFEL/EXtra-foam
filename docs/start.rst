GETTING STARTED
===============


Choose the correct version
--------------------------

**EXtra-foam** can be started on both online and `Maxwell` clusters. Currently, there
are two versions of **EXtra-foam** deployed. Please always consult your contact person
if you are not sure which version to use.


I. Latest version
+++++++++++++++++++++++

This is the latest release of **EXtra-foam**. This version usually contains more
features than the **stable** version.

.. code-block:: bash

    module load exfel EXtra-foam/beta
    extra-foam DETECTOR TOPIC

More info on command line arguments can be obtained as

.. code-block:: console

   [user@exflonc12 ~]$ extra-foam --help

    usage: extra-foam [-h] [-V] [--debug] [--redis_address REDIS_ADDRESS]
                      {AGIPD,LPD,DSSC,JUNGFRAUPR,JUNGFRAU,FASTCCD,EPIX100,BASLERCAMERA}
                      {SPB,FXE,SCS,SQS,MID,HED}

    positional arguments:
      {AGIPD,LPD,DSSC,JUNGFRAUPR,JUNGFRAU,FASTCCD,EPIX100,BASLERCAMERA}
                            detector name (case insensitive)
      {SPB,FXE,SCS,SQS,MID,HED}
                            Name of the instrument

    optional arguments:
      -h, --help            show this help message and exit
      -V, --version         show program's version number and exit
      --n_modules N_MODULES
                            Number of detector modules. It is only available for
                            using single-module detectors like JungFrau in a
                            combined way. Not all single-module detectors are
                            supported.
      --debug               Run in debug mode
      --pipeline_slow_policy {0,1}
                            Pipeline policy when the processing rate is slower
                            than the arrival rate (0 for always process the latest
                            data and 1 for wait until processing of the current
                            data finishes)
      --redis_address REDIS_ADDRESS
                            Address of the Redis server


For more details about detector modules, please refer to :ref:`Geometry`.

.. note::
    It sometime takes more than a minute to start **EXtra-foam** for the first time! This
    is actually an issue related to the infrastructure and not because
    **EXtra-foam** is slow.

.. note::
    If you are connecting to the online or `Maxwell` clusters via SSH, you will need
    to enable X11 forwarding by including the -X option.

.. note::
    In order to have a better experience with **EXtra-foam** on the `Maxwell` cluster,
    you should need FastX2_ at max-display_. There is also a link for downloading
    the desktop client on the bottom-right corner when you opened max-display_ and logged in.
    For more details, please refer to the official website for FastX2_ at DESY.

.. _FastX2: https://confluence.desy.de/display/IS/FastX2
.. _max-display: https://max-display.desy.de:3443/


II. Stable version
++++++++++++++++++

To start the **stable** version on online or `Maxwell` clusters:

.. code-block:: bash

    module load exfel EXtra-foam
    extra-foam DETECTOR TOPIC


III. Test version
++++++++++++++++++

To start the **test** version on online or `Maxwell` clusters:

.. code-block:: bash

    module load exfel EXtra-foam/alpha
    extra-foam DETECTOR TOPIC

.. note::
    **test** version is not covered by OCD!


Data analysis in real time
--------------------------

For real-time data analysis, the (calibrated) data is streamed via a `ZMQ bridge`, which is
a `Karabo` device (`PipeToZeroMQ`) running inside the control network.

.. image:: images/data_source_from_bridge.png
   :width: 500


.. _online-clusters: https://in.xfel.eu/readthedocs/docs/data-analysis-user-documentation/en/latest/computing.html#online-cluster

.. note::
    Please check the online-clusters_ available for users at different instruments.

.. note::
  The entire data analysis workflow with relevant hostnames and ports are provided in the instrument support
  `documentation <https://in.xfel.eu/readthedocs/docs/fxe-instrument-control-infrastructure/en/latest/fxe_dataanalysis_toolbox.html>`__

Data analysis with files
------------------------

See :ref:`stream data from run directory`
