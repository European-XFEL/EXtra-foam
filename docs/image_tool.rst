IMAGE TOOL
==========

The *ImageTool* window is the second control window which provides various image-related
information and controls.


Image control
_____________

.. image:: images/ImageTool.png
   :width: 800

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Update image*             | Manually update the current displayed image in the *ImageTool*     |
|                            | window. Disabled if *Update automatically* is checked.             |
+----------------------------+--------------------------------------------------------------------+
| *Update automatically*     | Automatically update the current displayed image in the            |
|                            | *ImageTool* window.                                                |
+----------------------------+--------------------------------------------------------------------+
| *Auto level*               | Update the detector images (not only in the *ImageTool* window,    |
|                            | but also in other plot windows) by automatically selecting levels  |
|                            | based on the maximum and minimum values in the data.               |
+----------------------------+--------------------------------------------------------------------+
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
| *Subtract dark*            | Apply pulse-by-pulse dark subtraction if checked.                  |
+----------------------------+--------------------------------------------------------------------+
| *Subtract background*      | A fixed background value to be subtracted from all the pixel       |
|                            | values.                                                            |
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


Mask image
""""""""""

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


ROI manipulation
""""""""""""""""

You can activate (tick **On**) up to 4 ROIs at the same time. One can change the size
(**w**\idth, **h**\eight) and position (**x**\, **y**\) of an ROI by either dragging and moving
the ROI on the image or entering numbers. You can avoid modifying an ROI unwittingly by
**Lock**\ing it.


ROI 1D projection
"""""""""""""""""

Define the 1D projection of ROI (region of interest) analysis setup.

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Direction*                | Direction of 1D projection (x or y).                               |
+----------------------------+--------------------------------------------------------------------+
| *Normalizer*               | Normalizer of the 1D-projection VFOM.                              |
+----------------------------+--------------------------------------------------------------------+
| *AUC range*                | AUC (area under a curve) integration range.                        |
+----------------------------+--------------------------------------------------------------------+
| *FOM range*                | Integration range when calculating the figure-of-merit of 1D       |
|                            | projection.                                                        |
+----------------------------+--------------------------------------------------------------------+


Dark run
________

Users can record a "dark run" whenever data is available. The dark run consists of a number
of trains. The moving average of the each "dark pulse" in the train will be calculated,
which will then be used to apply dark subtraction to image data pulse-by-pulse.

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Record dark*              | Start and stop dark run recording.                                 |
+----------------------------+--------------------------------------------------------------------+
| *Remove dark*              | Remove the recorded dark run.                                      |
+----------------------------+--------------------------------------------------------------------+

.. Note::

    The moving average here is not calculated by nanmean_, which means that if a pixel of the image
    in a certain pulse is *NaN*, the moving average of that pixel will be *NaN* for that pulse.


Azimuthal integration
_____________________

**EXtra-foam** uses pyFAI_ to do azimuthal integration. As illustrated in the sketch below,
the **origin** is located at the sample position, more precisely, where the X-ray beam crosses
the main axis of the diffractometer. The detector is treated as a rigid body, and its position
in space is described by six parameters: 3 translations and 3 rotations. The orthogonal
projection of **origin** on the detector surface is called **PONI** (Point Of Normal Incidence).
For non-planar detectors, **PONI** is defined in the plan with z=0 in the detector’s coordinate
system. It is worth noting that usually **PONI** is not the beam center on the detector surface.

The input parameters *Cx* and *Cy* correspond to *Poni2* and *Poni1* in the
aforementioned coordinate system, respectively.

.. image:: images/pyFAI_PONI.png
   :width: 800

.. image:: images/azimuthal_integ_1D.png
   :width: 800

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Cx (pixel)*               | Coordinate of the point of normal incidence along the detector's   |
|                            | 2nd dimension, in pixel.                                           |
+----------------------------+--------------------------------------------------------------------+
| *Cy (pixel)*               | Coordinate of the point of normal incidence along the detector's   |
|                            | 1st dimension, in pixel.                                           |
+----------------------------+--------------------------------------------------------------------+
| *Pixel x (m)*              | Pixel size along the detector's 2nd dimension, in meter.           |
+----------------------------+--------------------------------------------------------------------+
| *Pixel y (m)*              | Pixel size along the detector's 1st dimension, in meter.           |
+----------------------------+--------------------------------------------------------------------+
| *Sample distance*          | Sample-detector distance in m. Only used in azimuthal integration. |
+----------------------------+--------------------------------------------------------------------+
| *Photon energy*            | Photon energy in keV. Only used in azimuthal integration for now.  |
+----------------------------+--------------------------------------------------------------------+
| *Integ method*             | Azimuthal integration methods provided by pyFAI_.                  |
+----------------------------+--------------------------------------------------------------------+
| *Integ points*             | Number of points in the output pattern of azimuthal integration.   |
+----------------------------+--------------------------------------------------------------------+
| *Integ range*              | Azimuthal integration range, in 1/A.                               |
+----------------------------+--------------------------------------------------------------------+
| *Normalizer*               | Normalizer of the azimuthal integration result.                    |
+----------------------------+--------------------------------------------------------------------+
| *AUC range*                | AUC (area under a curve) range, in 1/A.                            |
+----------------------------+--------------------------------------------------------------------+
| *FOM range*                | Integration range when calculating the figure-of-merit of the      |
|                            | azimuthal integration result, in 1/A.                              |
+----------------------------+--------------------------------------------------------------------+


Geometry
________

Geometry is only available for the multi-module detector which requires a geometry file to
assemble the images from different modules, for example, AGIPD, LPD and DSSC. **EXtra-foam**
uses karabo_data_ for image assembling. For detailed information about geometries of those
detectors, please refer to
https://karabo-data.readthedocs.io/en/latest/geometry.html

+----------------------------+--------------------------------------------------------------------+
| Input                      | Description                                                        |
+============================+====================================================================+
| *Quadrant positions*       | The first pixel of the first module in each quadrant,              |
|                            | corresponding to data channels 0, 4, 8 and 12.                     |
+----------------------------+--------------------------------------------------------------------+
| *Load geometry file*       | Open a *FileDialog* window to choose a geometry file from the      |
|                            | local file system. For LPD and DSSC, **Extra-foam** provides a     |
|                            | default geometry file.                                             |
+----------------------------+--------------------------------------------------------------------+
