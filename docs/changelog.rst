CHANGELOG
=========

0.7.2 (16 January 2020)
-----------------------

- **Improvement**

    - Remove 'AZIMUTHAL_INTEG_RANGE' from configuration #32
    - Remove 'process monitor' from action and make it a tab in DataSourceWidget #32
    - Reduce the update frequency of plots which accumulates data, for example, correlation,
      histogram, heatmap, etc., to 1 Hz #31
    - Improve Redis server configuration #29
    - Allow ImageViewF.setImage(None) #28
    - Provide better interface for users to call C++ code #25
    - Log geometry change and remove 'AZIMUTHAL_INTEG_POINTS", "CENTER_X", "CENTER_Y" from
      configuration #24
    - Rearrange C++ code and separate benchmark code from unittest #15
    - Re-implement PairData -> SimplePairSequence and AccumulatedData -> OneWayAccuPairSequence #14
    - Re-implement BinProcessor. Now, data history is stored and users can re-bin it at anytime #14
    - Reduce MAX_QUEUE_SIZE from 5 to 2 to reduce latency #14
    - Remove 'update_hist' in PumpProbeData and CorrelationData. Now GUI update is completely
      decoupled from processors #14
    - Merge CorrelationWindow into StatisticsWindow. Rename the old statistics widgets to histogram
      widgets; add a new tab in the MainGUI which is dedicated for 'statistics' control #14
    - Update dependencies #11
    - Simplify ThreadLogger code #10

- **New Feature**

    - Implement q-map visualization #32
    - Implement pixel-wise gain-offset correction by loading numpy array from files #25
    - New ROI analysis interface (enable different FOMs of ROI; enable pulse-resolved
      ROI normalizer; enable pulse-resolved ROI1 +/- ROI2 FOM; enable visualization of
      ROI projection and pulse-resolved ROI FOM in ImageTool) #12

- **Bug Fix**

    - Fix a bug in MovingAverageScalar and MovingAverageArray. Setting a new
      value of None will reset the moving average instead of being ignored #14


0.7.1 (4 December 2019)
-----------------------

This is the first release after migrating from EuXFEL gitlab to github!!!

- **Improvement**

    - Rename omissive fai to foam and change config folder from karaboFAI to EXtra-foam #6

- **Test**
    - Migrate CI from EuXFEL gitlab to public github #1

0.7.0 (25 November 2019)
------------------------

- **Improvement**

    - Change supporting email, (long) description and header content in each file #174
    - Regularize Qt imports #173
    - Re-arange the GUI interface and move image related control into ImageTool #171
    - Add hiredis-py as dependency and improve redis connection infrastructure #170
    - Remove (canvas, dockarea, flowchart, multiprocess) from pyqtgraph code base #155

- **New Feature**

    - Support online FCCD raw data analysis #169
    - Publish available data sources in Redis and improve infrastructure in client proxy #166

- **Bug Fix**

    - Clean-up thread logger gracefully #170

0.6.2 (15 November 2019)
------------------------

- **Improvement**

    - Code clean up and improve base classes in GUI #164
    - Improve image processing code in cpp (align with xfai) #159
    - Enhance ImageTool interface (integrate functions in DarkRunWindow and OverviewWindow) #158

- **New Feature**

    - Introduce special analysis interface (implement tr-XAS) #165
    - Add an option to not normalize VFOM #162

- **Bug Fix**

    - Pulse slicer will also slice the stored dark images #165

0.6.1 (28 October 2019)
-----------------------

- **Improvement**

    - Remove XAS related code (GUI, processor, etc.) !154
    - Update import location of ZMQStreamer !151
    - Improve system information summary interface and enable detecting GPU resources !138

- **New Feature**

    - Implement normalization by XGM pipeline data !157
    - New data source management interface !157
    - Implemented web monitor in Dash !152

0.6.0 (31 August 2019)
----------------------

- **Bug Fix**

    - Assembling image from files, when non-detector source available in data !140
    - Add mid specific data sources in ctrl widget !139

- **Improvement**

    - Code clean-up ! 138
    - Remove moving average of images !128
    - Display number of filtered pulses/train in OverviewWindow !128
    - Raise StopPipelineError in ImageProcessorPulse instead of ProcessingError !128

- **New Feature**


- **Test**

0.5.5 (26 August 2019)
----------------------

- **Bug Fix**

    - Fix user defined control data in 1D binning analysis !134
    - Fix image mask in pulse-resolved ROI !133

- **Improvement**

    - Allow instrument sources to stream apart from DET !135
    - Allow shutdown idling karaboFAI instance remotely !130
    - Rearrange plot widgets !121
    - Improve the API for C++ image processing code !116 !129
    - AGIPD also works with bridge data with 'ONDA' format !115

- **New Feature**

    - Add statistics plot for pulse of interest !127

- **Test**

0.5.4 (20 August 2019)
----------------------

- **Bug Fix**

    - Fix bug if shape changes when using out array for assembling !122

- **Improvement**

- **New Feature**

    - Support pulse-resolved and two-module JungFrau !83

- **Test**

0.5.3 (16 August 2019)
----------------------

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

0.5.2 (9 August 2019)
---------------------

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

0.5.1 (5 August 2019)
---------------------

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
