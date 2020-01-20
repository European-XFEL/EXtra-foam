.. _Stream data from run directory:

STREAM DATA FROM RUN DIRECTORY
==============================


.. image:: images/data_source_from_file.png
   :width: 500


File streamer
"""""""""""""

**EXtra-foam** can be used to replay experiments with stored files. To start streaming,
click on the *Offline* window on the tool bar that opens the following window.

.. image:: images/file_stream_control.png

The run folder is browsed through the ``Load Run Folder`` button. The corrected image
data will be streamed from the run folder. If the run folder has path structure
as on `Maxwell GPFS` (/gpfs/exfel/exp/instrument/cycle/proposal/proc/runnumber) then once
the run folder is loaded, all the  slow/control sources available in the
corresponding *raw* folder (or same data folder if no corresponding raw
folder is found) are listed. Users can then choose slow data sources to stream
along with the fast image data.

The data is streamed from files after the ``Stream files`` button is clicked.

*Alternatively*, one can type

.. code-block:: bash

    extra-foam-stream DETECTOR_NAME port

in another terminal to open the above window. This is useful for development since one
does not have to set up the streamer again when restarting **EXtra-foam**.


Sample run directories
""""""""""""""""""""""

.. note::
    Streaming files from the online cluster is pretty fast. However, it is sometimes unbearable to stream
    a large run from the `Maxwell` cluster. For development, it is recommended to copy a few files in a run
    to a local directory.

+------------+---------------------------------------------------+------------------------------------------+
|            | Run directory                                     | Description                              |
+============+===================================================+==========================================+
| AGIPD      | /gpfs/exfel/d/raw/SPB/201931/p900086/r0009        | 250 pulses                               |
+------------+---------------------------------------------------+------------------------------------------+
| LPD        | /gpfs/exfel/exp/FXE/201802/p002218/raw/r0229      | ring, 100 pulses                         |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/FXE/201802/p002218/proc/r0229     | ring, 100 pulses                         |
+------------+---------------------------------------------------+------------------------------------------+
| DSSC       | /gpfs/exfel/exp/SCS/201901/p002212/raw/r0061      | pump-probe, 70 pulses                    |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SCS/201901/p002212/raw/r0059      | pump-probe (dark), 70 pulses             |
+------------+---------------------------------------------------+------------------------------------------+
| JungFrau   | /gpfs/exfel/exp/FXE/201930/p900063/proc/r1051     | pump-probe                               |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/FXE/201930/p900063/raw/r1051      | pump-probe                               |
+------------+---------------------------------------------------+------------------------------------------+
| FastCCD    | /gpfs/exfel/exp/SCS/201802/p002170/proc/r0141     |                                          |
|            +---------------------------------------------------+------------------------------------------+
|            | /gpfs/exfel/exp/SCS/201802/p002170/raw/r0141      |                                          |
+------------+---------------------------------------------------+------------------------------------------+
