.. _Analysis type:

Analysis type
=============

Each analysis will generate at least one FOM (figure-of-merit) and possibly also
a VFOM (vector figure-of-merit), which are typically used in :ref:`statistics analysis`.

**FOM**

A scalar which is used to characterize the performance of an analysis in a train/pulse.
For instance:

- (Sum, median, mean) of an ROI;
- (Absolute) sum of (part of) the difference curve in a pump-probe analysis;
- Absorption coefficient.

**VFOM**

An array of scalars which is used to characterize the performance of an analysis in a train/pulse.
For instance:

- Scattering curve from azimuthal integration;
- The difference curve in a pump-probe analysis;

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
     - Sum of the scattering curve, the value of the largest peak in the curve,
       :math:`q` of the largest peak, and the :math:`q` of the center of mass.
