PLOT WINDOWS
============

.. _nanmean: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html
.. _clipping: https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html
.. _imageio: https://github.com/imageio/imageio

One can open windows via the icons in the action bar of the MainGUI. Generally speaking, windows
serve different purposes: providing addition controls, grouping data visualization as well as
monitoring the internal status.

+------------------------------------------------+

Pump-probe window
-----------------

Dedicated window for pump-probe analysis.

+-------------------------------+---------------------------------+
| Average of all 'on' images    | Normalized 'on' and 'off' VFOMs |
+                               +---------------------------------+
|                               | Normalized 'on' - 'off' VFOM    |
+-------------------------------+---------------------------------+
| Average of all 'off'images    | FOM vs. train ID                |
+-------------------------------+---------------------------------+

Statistics window
-----------------

+------------------------------------------------+
| Pulse-resolved FOM in the current train        |
+------------------------------------------------+
| Histogram of pulse-/train- resolved FOM        |
+------------------------------------------------+


Correlation window
------------------

+-------------------------------+--------------------------------+
| Correlator 1 vs. FOM          | Correlator 2 vs. FOM           |
+-------------------------------+--------------------------------+
| Correlator 3 vs. FOM          | Correlator 4 vs. FOM           |
+-------------------------------+--------------------------------+


Binning 1D window
-----------------

+-----------------------------------------------------+------------------------------------------------------+
| VFOM vs. bin center 1 heatmap                       | VFOM vs. bin center 1 heatmap                        |
+-----------------------------------------------------+------------------------------------------------------+
| averaged/accumulated FOM vs. bin center 1 histogram | averaged/accumulated FOM vs. bin center 1 histogram  |
+-----------------------------------------------------+------------------------------------------------------+
| FOM count vs. bin center 1 histogram                | FOM count vs. bin center 1 histogram                 |
+-----------------------------------------------------+------------------------------------------------------+

Binning 2D window
-----------------

+----------------------------------------------------------------------------+
| averaged/accumulated FOM vs. bin center 1 (x) and bin center 2 (y) heatmap |
+----------------------------------------------------------------------------+
| FOM count heatmap                                                          |
+----------------------------------------------------------------------------+

Pulse-of-interest window
------------------------

+---------------+
| POI 1 image   |
+---------------+
| POI 2 image   |
+---------------+

Process monitor
---------------

Monitoring the status of processes running by **EXtra-foam**.


File streamer
-------------

A satellite control window which is used to stream image data together with slow data
from files.

.. image:: images/file_stream_control.png


.. _ImageFileFormat:

.. Note:: Image file format

    The two recommended image file formats are `.npy` and `.tif`. However,
    depending on the OS, the opened file dialog may allow you to enter any filename.
    Therefore, in principle, users can save and load any other image file formats
    supported by imageio_. However, it can be wrong if one writes and then loads a
    `.png` file due to the auto scaling of pixel values.
