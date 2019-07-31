karaboFAI
=========

karaboFAI is an application  that provides on-line (real-time, as fast as the
calibration pipeline) and off-line data analysis and visualization for experiments 
at European XFEL that using 2D detectors, e.g. LPD, JungFrau, etc.

[Documentation](https://in.xfel.eu/readthedocs/docs/karabofai/en/documentation/)

## Build and install

You are encouraged to use [Anaconda](https://www.anaconda.com/) to run and build **karaboFAI**.

### Dependencies

- cmake >= 3.8
- gcc >= 5.4 (support c++14)

In your [Anaconda](https://www.anaconda.com/) environment, run the following commands:

```sh
$ conda install -c anaconda cmake
$ conda install -c omgarcia gcc-6
```

### Install karaboFAI

```sh
$ git clone --recursive https://git.xfel.eu/gitlab/dataAnalysis/karaboFAI.git

# If you have cloned the repository without one or more of its submodules, run
$ git submodule update --init

$ cd karaboFAI

# optional
$ export FAI_WITH_TBB=0  # turn off TBB
$ export FAI_WITH_XSIMD=0  # turn off XSIMD

# Note: This step is also required if one wants to change the above 
#       environmental parameters.
$ python setup.py clean  # alternatively "rm -r build"

$ pip install .
```
