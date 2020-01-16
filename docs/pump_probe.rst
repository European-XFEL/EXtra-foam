MAIN GUI
========

.. _nanmean: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html


Pump-probe setup
""""""""""""""""

In the *pump-probe* analysis, the average (nanmean_) images of the on- and off- pulses are
calculated by

.. math::

   \bar{I}_{on} = \Sigma I_{on} / N_{on}

   \bar{I}_{off} = \Sigma I_{off} / N_{off} .

Then, moving averages of VFOM (on) and VFOM (off) for :math:`\bar{I}_{on}` and :math:`\bar{I}_{off}`
will be calculated, respectively, depending on the specified analysis type. The VFOM of *pump-probe*
analysis is given by VFOM (on) - VFOM (off).

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *On/off mode*              | Pump-probe analysis mode:                                          |
|                            |                                                                    |
|                            | - *predefined off*:                                                |
|                            |                                                                    |
|                            |   On-pulses will be taken from each train while the 'off'          |
|                            |   (reference image) is specified in the ImageTool.                 |
|                            |                                                                    |
|                            | - *same train*:                                                    |
|                            |                                                                    |
|                            |   On-pulses and off-pulses will be taken from the same train. Not  |
|                            |   applicable to train-resolved detectors.                          |
|                            |                                                                    |
|                            | - *even\/odd*:                                                     |
|                            |                                                                    |
|                            |   On-pulses will be taken from trains with even train IDs while    |
|                            |   off-pulses will be taken from trains with odd train IDs.         |
|                            |                                                                    |
|                            | - *odd\/even*:                                                     |
|                            |                                                                    |
|                            |   On-pulses will be taken from trains with odd train IDs while     |
|                            |   off-pulses will be taken from trains with even train IDs.        |
+----------------------------+--------------------------------------------------------------------+
| *Analysis type*            | See AnalysisType_.                                                 |
+----------------------------+--------------------------------------------------------------------+
| *On-pulse indices*         | Indices of all on-pulses. **If 'Pulse slicer' is used to slice a   |
|                            | portion of the pulses in the train, these indices are indeed the   |
|                            | indices of the pulse in the sliced train**.                        |
|                            | *Pulse-resolved detector only.*                                    |
+----------------------------+--------------------------------------------------------------------+
| *Off-pulse indices*        | Indices of all off-pulses. *Pulse-resolved detector only.*         |
+----------------------------+--------------------------------------------------------------------+
| *FOM from absolute on-off* | If this checkbox is ticked, the FOM will be calculated based on    |
|                            | `\|on - off\|` (default). Otherwise `on - off`.                    |
+----------------------------+--------------------------------------------------------------------+
| Reset                      | Reset the FOM plot in the *Pump-probe window* and the global       |
|                            | moving average count.                                              |
+----------------------------+--------------------------------------------------------------------+

Pump-probe window
"""""""""""""""""

Dedicated window for pump-probe analysis.

+-------------------------------+---------------------------------+
| Average of all 'on' images    | Normalized 'on' and 'off' VFOMs |
+                               +---------------------------------+
|                               | Normalized 'on' - 'off' VFOM    |
+-------------------------------+---------------------------------+
| Average of all 'off'images    | FOM vs. train ID                |
+-------------------------------+---------------------------------+