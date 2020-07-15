CHANGELOG
=========

0.9.1 (15 July 2020)
------------------------

- **Bug Fix**

  - Fix transform type update in TransformView in ImageTool. #251

- **Improvement**

  - PlotItem will not be shown in legend if its name is empty. #254
  - Improve y2-axis plot implementation. #253
  - Improve stack detector modules. #247
  - Implement ScatterPlotItem to replace the pyqtgraph one. #238

- **New Feature**

  - Implement curve fitting in correlation window. #255
  - Implement azimuthal integration and concentric ring detection in C++. #252


0.9.0 (30 June 2020)
------------------------

- **Bug Fix**

  - Fix bug in data source tree introduced in 0.8.4. #235
  - Fix transparent Nan pixel with the latest pyqtgraph version. #231

- **Improvement**

  - Improve performance of PlotCurveItem and add benchmarks for PlotItems and ImageViewF. #243
  - Make ctrl widgets in Imagetool more compact. #241
  - Improve C++ code quality and suppress TBB deprecate warning. #239
  - Add meter bar to plot area. #236
  - Reimplement ImageItem and PlotItem to replace the pyqtgraph ones. #232 #242
  - Improve data source updating and color encoding matched source items in
    the data source tree. #230 #234
  - Rename configurator to analysis setup manager and allow take snapshot for
    each setup. #228
  - Update data source and trouble shooting in documentation. #227
  - Add summary of compiler flags for EXtra-foam-python. #226
  - Update installation (gcc7) and CI (gcc8). #226
  - Rename misleading mouse mode in the right-click context menu. #211
  - Update pyqtgraph to the latest master branch. #206 #211 #233

- **New Feature**

  - Annotate peaks in azimuthal integration view. #240
  - Enable Legend for all the plot items. #206 #237
  - Implement logarithmic X/Y scale in right-click context menu. #206
  - Enable C++ API installation and add examples. #227


0.8.4 (8 June 2020)
------------------------

- **Bug Fix**

  - Beam size in the bar plot will change when the resolution changes in the
    correlation analysis. #209

- **Improvement**

  - Update Redis versions: server -> 6.0.5; redis-py -> 3.5.2; hiredis-py -> 1.0.1. #220
  - Slightly improve image proc C++ code performance on machines with few threads. #219
  - Visualize data type in data source Tree. #218
  - Improve the performance of pulse filter. #214
  - Improve setup.py. #212
  - Keep data when resolution changes in correlation analysis; move sequence classes
    into algorithms/data_structures. #209
  - Mask ASIC edge when n_modules == 1; simplify geometry binding code. #207
  - Add benchmark for generalized geometry. #205
  - Make special suite more self-contained. #204
  - Mask JungFrau and JungFrauPR. #200
  - Move statistics ctrl widgets into corresponding windows. #199

- **New Feature**

  - Color encoding matched sources in the data source tree. #220
  - Add processed pulse counter in ImageTool. #216
  - Add support for C++ API. #213
  - Add xtensor-blas as a submodule. #210
  - Implement image transform processor and view (FFT, edge detection). #203
  - Integrate Karabo gate (PipeToEXtraFoam device) which allows request pipeline
    and control data in special suite. #168


0.8.3 (11 May 2020)
------------------------

- **Breaking change**
    - In the terminal, "--n_modules 2" is required to run JungFrauPR with two modules. #41

- **Bug Fix**
    - Change pixel size of ePix100 from 0.11 mm to 0.05 mm. #189

- **Improvement**
    - Mask tile/ASIC edges by default. #192
    - Improve geometry 1M and its unittest. #190
    - Invert y axis for displayed image. #187
    - Rename geometry to geometry_1m in C++. #186
    - Improve tr-XAS analysis in special suite. #163 #183
    - Improve correlating error message. #182
    - Improve documentation for special suite. #177
    - New reset interface in special suite. #170
    - Regularize names of methods and attributes in special suite. #167
    - Add new mode, start/end train ID control and progress bar, etc. in FileStreamer. #166
    - Move definition of meta source from config to SourceCatalog. #165
    - Use correlated queue in special suite. #164
    - Improve shape comparing error message in C++. #160
    - Improve mask image data implementation and interface. #157
    - Move image assembler into image processor. # 155
    - Refactor masking code. #149
    - Implement generic binding for nansum and nanmean. #114

- **New Feature**
    - Add axis calibration in Gotthard analysis. #179
    - Implement generalized geometry for multi-module detectors. #175 #196
    - Implement streaming JungFrauPR data from files. #174
    - Implement Gotthard pump-probe analysis in special suite. #173 #178
    - Add ROI histogram in CameraView in special suite. #172
    - Add ROI control in special suite. #171
    - Implement XAS-TIM-XMCD in special suite. #162
    - Implement MultiCameraView in special suite. #147
    - Implement XAS-TIM in special suite. #146
    - Implement load and save mask in pixel coordinates. #132 #154 #185 #191 #197


0.8.2 (8 April 2020)
------------------------

- **Bug Fix**

    - Fix not able to close file stream process when closing, if the file stream window
      is opened through the main GUI. #122
    - Fix offset correction switch between dark and offset. #141

- **Improvement**

    - Move mouse hover (x, y, v) display implementation to ImageViewF. #148
    - Visualize dark and offset separately. #141
    - Improve loading reference image and calibration constants. #141
    - Implement smart auto levels of image. #138
    - Enhance SourceCatalog.add_item. #137
    - Improve class init with moving average descriptor. #136
    - Bump EXtra-data version and remove duplicated code. #131
    - Tweak assembling code in C++ to make the result exactly the same as EXtra-geom. #129
    - Simplify ImageProc binding code. #125
    - Update dependencies. #118
    - Update documentation. #115 #130
    - Move tr-XAS analysis to special suite. #89

- **New Feature**

    - Generalize file stream. #122
    - Add standard deviation, variance and speckle contrast into ROI FOM. #119
    - Implement tile edge mask for modular detectors. #110
    - Add support for fast ADC as a digitizer source. #101
    - Implement Camera view (special suite). #89
    - Implement Gotthard analysis (special suite) for MID. #89
    - Implement interface and examples for special analysis suite. #89

0.8.1 (16 March 2020)
------------------------

- **Improvement**

    - Automatically reset empty image mask with inconsistent shape. #104

- **New Feature**

    - Implement AGIPD 1M geometry in C++. #102
    - Add ROI1_DIV_ROI2 as an option for ROI FOM. #103
    - Implement normalization for ROI FOM. #96
    - Implement ROI FOM master-slave scan. #93
    - Add branch-based CI and Singularity image deployment. #92
    - Add support for ePix100 detector. #90
    - Implement save and load metadata. #87

0.8.0.1 (3 March 2020)
------------------------

- **Bug Fix**

    - Fix display bug in ImageTool #85


0.8.0 (2 March 2020)
------------------------

- **Improvement**

    - Get rid of the artifact induced by masking pixel to zero when calculating
      statistics, e.g. mean, median, std.
    - Provide a mask to pyFAI to perform azimuthal integration. #61
    - New C++ implementation to mask pixel in Nan and/or return a boolean mask. #61
    - ROI pulse FOM and NORM will only be calculated after registration. #61

- **New Feature**

    - Enable train-resolved FOM filter. #78
    - Display numbers of processed and dropped trains. #77
    - Support online single module data from a modular detector. #72
    - Allow type selection for 1D projection (sum or mean). #71
    - Implement mouse cursor value indicator for PlotWidgetF. #66
    - Preliminary implementation of nanmean and nansum in C++. #61

- **Bug Fix**

    - Fix pulse-filter in digitizer. #80
    - Fix gain/offset slicer for train-resolved detectors. #76
    - Use nansum in Tr-XAS analysis. #75
    - Fix typo in unittest. #74
    - Fix changing device ID in data source on the fly. #69

0.7.3 (24 February 2020)
------------------------

- **Breaking change**

    - In the terminal, "--topic" becomes a positional argument. #41

- **Improvement**

    - Reimplement Color classes. mkPen and mkBrush from pyqtgraph are not needed
      anymore. #53
    - Allow select pipeline policy (wait or drop) via commandline. The default is wait
      since the data arrival speed is slower than the processing speed during online
      analysis. #45
    - Replace Python's build-in queue.Queue to speed up data transfer. #45
    - Improve the visualization of heatmap. #44
    - Allow starting instances with different detectors without warning message. #41
    - Allow to shutdown others' Redis server to avoid zombie Redis server occupying
      the port. #41
    - Implement Fast assembling for LPD and DSSC in C++. #40
    - Resign the config code. Now each instrument will has its own config file,
      e.g. scs.config.yaml, fxe.config.yaml. All the instrument sources will be
      set up in the config file. #38
    - Implement streaming raw (AGIPD, LPD) data from files and also 'confirmed'
      streaming raw (AGIPD, LPD) data online. #38

- **New Feature**

    - Allow specific bin range of histogram. #56
    - Provide ROI histogram for train-resolved detectors; Provide ROI histogram for
      the averaged image of pulse-resolved detectors. #56
    - Display `mean`, `median` and `std` for all histogram plots. #56
    - ROI histogram for pulse-resolved detectors. #55
    - Double-y plot for 1D binning. #53
    - Support normalizing by digitizer (TIM). #52
    - Support multiple ZMQ endpoints connections. #45
    - Automatically correlate data from the same/different endpoints with train ID. #45
    - Allow automatically choosing bin range. #44
    - Also add an option to stack the detectors (LPD and DSSC) without assembling. #40
    - Control required sources in the DataSourceTree. #38
    - Allow filtering by value for all non-detector data sources. #38
    - Implement AdqDigitizer processor. #38

- **Bug Fix**

    - Fix default AGIPD geometry. #62
    - Disable pulse slicer for train-resolved detectors in DataSourceTree and gain/offset
      correction. #56
    - Fix logger level. #41
    - Fix extra-foam-kill. #41

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
