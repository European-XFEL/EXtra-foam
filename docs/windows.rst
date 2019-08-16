WINDOWS
=======

.. _nanmean: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html
.. _clipping: https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html
.. _imageio: https://github.com/imageio/imageio

One can open windows via the icons in the action bar of the MainGUI. Generally speaking, windows
serve different purposes: providing addition controls, grouping data visualization as well as
monitoring the internal status.


ImageTool
---------

The *ImageTool* is the second control window which provides various operations on images.

.. image:: images/ImageTool.png
   :width: 800

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Moving average*           | Moving average window size of image data (only applied to          |
|                            | train-resolved detectors). It is worth noting that the average is  |
|                            | **not** calculated by nanmean_. It is helpful for visualization    |
|                            | if the signal is very weak. Please note that this moving average   |
|                            | will also affect the analysis. Namely, if this one and the moving  |
|                            | average in the *Global setup* are specified at the same time, you  |
|                            | will get a moving average of a moving average.                     |
+----------------------------+--------------------------------------------------------------------+
| *Threshold mask*           | An interval that pixel values outside the interval are set to 0.   |
|                            | Please distinguish *threshold mask* from clipping_.                |
+----------------------------+--------------------------------------------------------------------+
| *Subtract background*      | A fixed background value to be subtracted from all the pixel       |
|                            | values.                                                            |
+----------------------------+--------------------------------------------------------------------+

The action bar provides several actions for real-time masking operation. The pixel values in the
masked region will be set to 0.

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Mask*                     | Mask a rectangular region.                                         |
+----------------------------+--------------------------------------------------------------------+
| *Unmask*                   | Remove mask in a rectangular region.                               |
+----------------------------+--------------------------------------------------------------------+
| *Trash mask*               | Remove all the mask.                                               |
+----------------------------+--------------------------------------------------------------------+
| *Save image mask*          | Save the current image mask in `.npy` format.                      |
+----------------------------+--------------------------------------------------------------------+
| *Load image mask*          | Load a image mask in `.npy` format.                                |
+----------------------------+--------------------------------------------------------------------+

You can activate (tick **On**) up to 4 ROIs at the same time. One can change the size
(**w**\idth, **h**\eight) and position (**x**\, **y**\) of an ROI by either dragging and moving
the ROI on the image or entering numbers. You can avoid modifying an ROI unwittingly by
**Lock**\ing it.


Other buttons in the *ImageTool* window:

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Update image*             | Update the current displayed image in the *ImageTool* window.      |
+----------------------------+--------------------------------------------------------------------+
| *Auto level*               | Update the detector images (not only in the *ImageTool* window,    |
|                            | but also in other PlotWindows) by automatically selecting levels   |
|                            | based on the maximum and minimum values in the data.               |
+----------------------------+--------------------------------------------------------------------+
| *Save image*               | Save the current image to file. Please also see ImageFileFormat_   |
+----------------------------+--------------------------------------------------------------------+
| *Load reference*           | Load a reference image from file. Please also see ImageFileFormat_ |
+----------------------------+--------------------------------------------------------------------+
| *Set reference*            | Set the current displayed image as a reference image. For now,     |
|                            | reference image is used as a stationary off-image in the           |
|                            | *predefined off* mode in *pump-probe* analysis.                    |
+----------------------------+--------------------------------------------------------------------+
| *Remove reference*         | Remove the reference image.                                        |
+----------------------------+--------------------------------------------------------------------+


Overview window
---------------

+------------------------------------------------+
| Basic information of the current train         |
+------------------------------------------------+
| Average of all the images within the train     |
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

Azimuthal integration window
----------------------------

+---------------------------------------------------------------------------+
| Azimuthal integration of the average of all pulses in the current train   |
+---------------------------------------------------------------------------+

Process monitor
---------------

Monitoring the status of processes running by **karaboFAI**.


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
