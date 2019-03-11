Introduction
============

**karaboFAI** is a tool that provides real-time and off-line data analysis
(azimuthal integration, ROI, correlation, etc.) and visualization for
experiments using 2D detectors at European XFEL.
The current version works with AGIPD, LPD, JungFrau and FastCCD.

.. list-table:: Performance (ms/train)
   :header-rows: 1

   * - Detector
     - With azimuthal integration
     - Without azimuthal integration

   * - AGIPD (60 pulses/train)
     - ~ 2200
     - ~ 700
   * - LPD (60 pulses/train)
     - ~ 2200
     - ~ 700
   * - Jungfrau
     - < 200
     - < 10
   * - FastCCD
     - ~ 500
     - < 20


.. image:: karaboFAI-GUI.png
   :width: 800

.. image:: karaboFAI-ImageTool.png
   :width: 800

.. image:: karaboFAI-LPD_azimuthal_integration.png
   :width: 800

.. image:: karaboFAI-ROI.png
   :width: 800