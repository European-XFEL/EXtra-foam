Graphical User Interface (GUI)
==============================

The main GUI of **karaboFAI** is divided into several control panels grouped
by functionality and a log window.


General analysis setup panel
----------------------------

Define the general analysis setup.

Azimuthal integration setup panel
---------------------------------

Define the azimuthal integration setup which will be passed to **pyFAI**.

Correlation analysis setup panel
--------------------------------

Define the figure of merit (FOM) and the correlated parameters.

Pump-probe analysis setup panel
-------------------------------

Configure the parameters which is used in pump-and-probe experiments
with an optical laser. If activate, **karaboFAI** calclates:

- The moving average of the average of the azimuthal integration
  of all laser-on and laser-off pulses, as well as their difference;
- The evolution of the figure of merit (FOM), which is integration
  of the absolute difference between the moving average of the
  laser-on and laser-off results, for each pair of laser-on and
  laser-off trains.


Geometry setup panel
--------------------

Geometry setup panel is only available for the detector which requires a
geometry file to assembled the images from different modules, for example,
AGIPD and LPD.


Data source setup panel
-----------------------

