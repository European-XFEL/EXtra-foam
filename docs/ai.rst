Azimuthal Integration
=====================

**karaboFAI** uses pyFAI_ to do azimuthal integration. As illustrated in the
sketch below, the **origin** is located at the sample position, more precisely,
where the X-ray beam crosses the main axis of the diffractometer. The detector
is treated as a rigid body, and its position in space is described by six
parameters: 3 translations and 3 rotations. The orthogonal projection of
**origin** on the detector surface is called **PONI** (Point Of Normal
Incidence). For non-planar detectors, **PONI** is defined in the plan with z=0
in the detector’s coordinate system. It is worth noting that usually **PONI**
is not the beam center on the detector surface.

The input parameters *Cx* and *Cy* correspond to *Poni2* and *Poni1* in the
aforementioned coordinate system, respectively.

.. _pyFAI: https://github.com/silx-kit/pyFAI

.. image:: images/pyFAI_PONI.png
   :width: 800
