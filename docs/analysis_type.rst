.. _Analysis type:

Analysis type
=============

Each analysis type starts from an (ROI) image and will generate a FOM (figure-of-merit) and a VFOM
(vector figure-of-merit). Take the analysis type *ROI (proj)* for example, it starts from the image
which is the subtraction of ROI1 and ROI2. The VFOM is the projection of this image in the x or y
direction, and the FOM the sum of the absolute VFOM.

.. list-table::
   :header-rows: 1

   * - Type
     - Description
     - VFOM
     - FOM

   * - *pump-probe*
     - See :ref:`PUMP-PROBE ANALYSIS`.
     - VFOM (on) minus VFOM (off).
     - Sum of the (absolute) on-off VFOM.

   * - *ROI*
     - Region of interest.
     - NA
     - Defined in :ref:`ROI FOM setup`.

   * - *ROI (proj)*
     - 1D projection in x/y direction of ROI.
     - Projection of ROI in the x/y direction.
     - Defined in :ref:`ROI projection setup`.

   * - *azimuthal integ*
     - 1D Azimuthal integration of average (pulse) image(s) in a train.
     - Azimuthal integration scattering curve.
     - Sum of the scattering curve.
