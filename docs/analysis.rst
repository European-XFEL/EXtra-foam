Data Analysis with karaboFAI
============================


Real-time data analysis
#######################

To start `karaboFAI` on any online cluster:

.. code-block:: bash

    /gpfs/exfel/sw/software/karaboFAI/env/bin/karaboFAI DETECTOR_NAME


Valid detectors are `AGIPD`, `LPD` and `JungFrau`.

.. note::
   It usually takes a long time to start `karaboFAI` for the first time! This is actually an issue related to the infrastructure and not because `karaboFAI` is slow.


.. list-table:: Suggested online clusters
   :header-rows: 1

   * - Instrument
     - Alias
     - DNS primary name

   * - SPB
     - sa1-br-onc-comp-spb
     - exflonc05
   * - FXE
     - sa1-br-onc-comp-fxe
     - exflonc12
   * - SCS
     - sa1-br-kc-comp-1
     - exflonc13
   * - SQS
     - sa1-br-kc-comp-3
     - exflonc15

Off-line data analysis
######################

For now, `karaboFAI` can be used to replay the experiment with files.

The way to start `karaboFAI` on `Maxwell cluster` is the same as on the `online cluster`:

.. code-block:: bash

    /gpfs/exfel/sw/software/karaboFAI/env/bin/karaboFAI DETECTOR_NAME

.. list-table:: Example files
   :header-rows: 1

   * - Detector
     - File directory

   * - AGIPD
     - /gpfs/exfel/exp/XMPL/201750/p700000/proc/r0273
   * - LPD
     - /gpfs/exfel/exp/FXE/201701/p002026/proc/r0078
   * - JungFrau
     - /gpfs/exfel/exp/FXE/201801/p002118/proc/r0143
   * - FastCCD
     -
