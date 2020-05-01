.. _stream data from run directory:

STREAM DATA FROM RUN DIRECTORY
==============================


.. image:: images/data_source_from_file.png
   :width: 500


File stream
"""""""""""

**EXtra-foam** can be used to replay experiments with stored files. To start,
click on the *File stream* icon on the tool bar to opens the following window:

.. image:: images/file_stream.png

*Alternatively*, one can type

.. code-block:: bash

    extra-foam-stream

in another terminal to open the above window. This is useful for development since one
does not have to set up the streamer again when restarting **EXtra-foam**.

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| ``Load run``               | Click to select a run folder. The run folder can also be specified |
|                            | via entering the full path.                                        |
+----------------------------+--------------------------------------------------------------------+
| ``Port``                   | The TCP port from which the data is streamed. If the GUI is not    |
|                            | opened from the terminal, the ``Port`` is readonly and internally  |
|                            | it is the same as the port specified in the :ref:`Data source`     |
|                            | panel in the main GUI.                                             |
+----------------------------+--------------------------------------------------------------------+
| ``Stream once``            | Press to stream the data in the run folder once.                   |
+----------------------------+--------------------------------------------------------------------+
| ``Stream repeatedly``      | Press to stream the data in the run folder repeatedly.             |
|                            | If the stream reaches the end of the data, the                     |
|                            | stream will restart from the beginning with a faked *train ID*,    |
|                            | which ensures that the *train ID* continuously increases in the    |
|                            | new cycle. This feature is only useful for developers.             |
+----------------------------+--------------------------------------------------------------------+
| ``Stop stream``            | Press to stop streaming.                                           |
+----------------------------+--------------------------------------------------------------------+
| ``Mode``                   | Stream mode:                                                       |
|                            |                                                                    |
|                            | - ``Normal``:                                                      |
|                            |                                                                    |
|                            |   Sources in a train are streamed together.                        |
|                            |                                                                    |
|                            | - ``Random shuffle``:                                              |
|                            |                                                                    |
|                            |   Sources in a train are streamed one by one and the order is      |
|                            |   random.                                                          |
+----------------------------+--------------------------------------------------------------------+
| ``First train ID``         | Slide to change the first train ID to stream.                      |
+----------------------------+--------------------------------------------------------------------+
| ``Last train ID``          | Slide to change the last train ID to stream.                       |
+----------------------------+--------------------------------------------------------------------+
| ``Stride``                 | Train ID stride used to slice trains to stream.                    |
+----------------------------+--------------------------------------------------------------------+

.. note::
    If the specified run folder has a path structure as on `Maxwell GPFS` (*.../proc/runnumber/*).
    The loader will try to load the data (e.g. control data) other than the
    detector data from *.../raw/runnumber/* first and fall back to *.../proc/runnumber/* if
    *.../raw/runnumber/* is not a valid run directory. This is needed for large multi-module
    detectors, as the folder *.../proc/runnumber/* only stores the calibrated detector data.

Sample run directories
""""""""""""""""""""""

.. note::
    Streaming files from the online cluster is pretty fast. However, it is sometimes unbearable to stream
    a large run from the `Maxwell` cluster. For development, it is recommended to copy a few files in a run
    to a local directory.

+------------+---------------------------------------------------+------------------------------------------+
|            | Run directory                                     | Description                              |
+============+===================================================+==========================================+
| AGIPD      | /gpfs/exfel/exp/SPB/201831/p900039/proc/r0273     | ring; 176 pulses                         |
+------------+---------------------------------------------------+------------------------------------------+
| LPD        | /gpfs/exfel/exp/FXE/201802/p002218/raw/r0229      | ring, 100 pulses                         |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/FXE/201802/p002218/proc/r0229     | ring, 100 pulses                         |
+------------+---------------------------------------------------+------------------------------------------+
| DSSC       | /gpfs/exfel/exp/SCS/201901/p002212/raw/r0061      | pump-probe; 70 pulses                    |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SCS/201901/p002212/raw/r0059      | pump-probe (dark of r0061)               |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SCS/201901/p002161/raw/r0093      | tr-XAS, single module, 50 pulses         |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SCS/201901/p002161/raw/r0095      | tr-XAS (dark of r0093)                   |
+------------+---------------------------------------------------+------------------------------------------+
| JungFrau   | /gpfs/exfel/exp/FXE/201930/p900063/proc/r1051     | pump-probe                               |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/FXE/201930/p900063/raw/r1051      | pump-probe                               |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SPB/201922/p002566/proc/r0061     | Burst mode; ring; 6 modules              |
+------------+---------------------------------------------------+------------------------------------------+
| FastCCD    | /gpfs/exfel/exp/SCS/201802/p002170/proc/r0141     |                                          |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SCS/201802/p002170/raw/r0141      |                                          |
+------------+---------------------------------------------------+------------------------------------------+
| Gotthard   | /gpfs/exfel/exp/MID/201931/p900090/raw/r0395      | Test data                                |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/MID/201931/p900090/raw/r0300      | Test data (dark)                         |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SCS/201931/p900094/raw/r0647      | pump-probe                               |
+------------+---------------------------------------------------+------------------------------------------+
| XAS-TIM    | /gpfs/exfel/exp/SCS/201931/p900094/raw/r0491      | XMCD, 42 pulses/train, APD stride = 1    |
+------------+---------------------------------------------------+------------------------------------------+
