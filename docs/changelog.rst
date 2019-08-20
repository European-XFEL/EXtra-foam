CHANGELOG
=========

karaboFAI 0.5.4 (20 August 2019)
---------------------------------

- **Bug Fix**

    - Fix bug if shape changes when using out array for assembling !122

- **Improvement**

- **New Feature**

    - Support pulse-resolved and two-module JungFrau !83

- **Test**

karaboFAI 0.5.3 (16 August 2019)
---------------------------------

- **Bug Fix**

    - Fix series nan mean two images !106

- **Improvement**

    - Introduce 'TOPIC' to separate instrument specific sources !114
    - Implement masking image in cpp !110

- **New Feature**

    - Implement DarkRunWindow !109
    - Allow save image and load reference in ImageTool !107

- **Test**

    - Integrate cpp unittest into setuptools and CI (both parallel and series) !110

karaboFAI 0.5.2 (9 August 2019)
-------------------------------

- **Bug Fix**

- **Improvement**

    - Prevent costly GUI updating from blocking data acquisition !101
    - Improve nanmean performance when simple slice is not applicable !97
    - Add output array in image assembly !85

- **New Feature**

    - List critical information of a run in FileStreamer window !103
    - Implement AboutWindow !102
    - Pulse slicing and data reduction !99
    - New widget SmartSliceLineEdit !98

- **Test**


karaboFAI 0.5.1 (5 August 2019)
-------------------------------

- **Bug Fix**

    - Capture exception when trying to kill others' instance !93
    - Add AGPID detector in FileServer !90
    - Fix when a new detector key cannot be found in an old config file !87

- **Improvement**

    - Implement parallel version of xt_nanmean_images !91
    - Delete detector data in raw data after Assembling !88
    - Update geometry file and default quad positins for DSSC !86
    - Make compiling with TBB and XSIMD default !84

- **New Feature**

    - Added MID_DET... source to list in AGIPD dict in config.py !94

- **Test**

    - Unittest statistics #82
    - Unittest for command proxy #81
