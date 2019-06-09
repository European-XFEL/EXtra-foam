karaboFAI
=========

karaboFAI is a tool that provides on-line (real-time, as fast as the
calibration pipeline) and off-line data analysis and visualization
for experiments at European XFEL that require azimuthal integration
of diffraction data acquired with 2D detectors.

[Documentation](https://in.xfel.eu/readthedocs/docs/karabofai/en/documentation/)

## Build and install

```sh
$ git clone --recursive https://git.xfel.eu/gitlab/dataAnalysis/karaboFAI.git
$ cd karaboFAI

# optional
$ export FAI_WITH_TBB=1  # libtbb-dev is required
$ export FAI_WITH_XSIMD=1

# Note: This step is also required if one wants to change the above 
#       environmental parameters.
$ python setup.py clean  # alternatively "rm -r build"

$ pip install .
```
