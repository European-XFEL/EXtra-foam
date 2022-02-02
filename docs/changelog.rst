Changelog
=========

1.0.0 (31 July 2020)
------------------------

- **Improvement**

    - Improve FFT of image in feature extraction. :ghpull:`268`
    - Improve fitting interface. :ghpull:`265`
    - Add reset panel in the main GUI. :ghpull:`264`
    - Update default config files. :ghpull:`262`
    - Improve azimuthal integration performance. :ghpull:`261`, :ghpull:`268`
    - Revert y-axis orientation change. :ghpull:`260`
    - Add Python binding for double dtype in generalized geometry. :ghpull:`259`

- **New Feature**

    - Implemented pulse-resolved correlation in special suite. :ghpull:`270`
    - Add curve fitting in histogram window. :ghpull:`267`
    - Support streaming Basler camera from files. :ghpull:`266`
    - Add more fitting types, Gaussian, Lorentzian and erf. :ghpull:`265`
    - Add pump-probe FOM to histogram window. :ghpull:`263`
    - Reset moving average in correlation scan mode. :ghpull:`263`


0.9.1 (15 July 2020)
------------------------

- **Bug Fix**

    - Fix transform type update in TransformView in ImageTool. :ghpull:`251`

- **Improvement**

    - PlotItem will not be shown in legend if its name is empty. :ghpull:`254`
    - Improve y2-axis plot implementation. :ghpull:`253`
    - Improve stack detector modules. :ghpull:`247`
    - Implement ScatterPlotItem to replace the pyqtgraph one. :ghpull:`238`

- **New Feature**

    - Implement curve fitting in correlation window. :ghpull:`255`
    - Implement azimuthal integration and concentric ring detection in C++. :ghpull:`252`


0.9.0 (30 June 2020)
------------------------

- **Bug Fix**

    - Fix bug in data source tree introduced in 0.8.4. :ghpull:`235`
    - Fix transparent Nan pixel with the latest pyqtgraph version. :ghpull:`231`

- **Improvement**

    - Improve performance of PlotCurveItem and add benchmarks for PlotItems and ImageViewF. :ghpull:`243`
    - Make ctrl widgets in Imagetool more compact. :ghpull:`241`
    - Improve C++ code quality and suppress TBB deprecate warning. :ghpull:`239`
    - Add meter bar to plot area. :ghpull:`236`
    - Reimplement ImageItem and PlotItem to replace the pyqtgraph ones. :ghpull:`232`, :ghpull:`242`
    - Improve data source updating and color encoding matched source items in
      the data source tree. :ghpull:`230`, :ghpull:`234`
    - Rename configurator to analysis setup manager and allow take snapshot for
      each setup. :ghpull:`228`
    - Update data source and trouble shooting in documentation. :ghpull:`227`
    - Add summary of compiler flags for EXtra-foam-python. :ghpull:`226`
    - Update installation (gcc7) and CI (gcc8). :ghpull:`226`
    - Rename misleading mouse mode in the right-click context menu. :ghpull:`211`
    - Update pyqtgraph to the latest master branch. :ghpull:`206`, :ghpull:`211`, :ghpull:`233`

- **New Feature**

    - Annotate peaks in azimuthal integration view. :ghpull:`240`
    - Enable Legend for all the plot items. :ghpull:`206`, :ghpull:`237`
    - Implement logarithmic X/Y scale in right-click context menu. :ghpull:`206`
    - Enable C++ API installation and add examples. :ghpull:`227`


0.8.4 (8 June 2020)
------------------------

- **Bug Fix**

    - Beam size in the bar plot will change when the resolution changes in the
      correlation analysis. :ghpull:`209`

- **Improvement**

    - Update Redis versions: server -> 6.0.5; redis-py -> 3.5.2; hiredis-py -> 1.0.1. :ghpull:`220`
    - Slightly improve image proc C++ code performance on machines with few threads. :ghpull:`219`
    - Visualize data type in data source Tree. :ghpull:`218`
    - Improve the performance of pulse filter. :ghpull:`214`
    - Improve setup.py. :ghpull:`212`
    - Keep data when resolution changes in correlation analysis; move sequence classes
      into algorithms/data_structures. :ghpull:`209`
    - Mask ASIC edge when n_modules == 1; simplify geometry binding code. :ghpull:`207`
    - Add benchmark for generalized geometry. :ghpull:`205`
    - Make special suite more self-contained. :ghpull:`204`
    - Mask JungFrau and JungFrauPR. :ghpull:`200`
    - Move statistics ctrl widgets into corresponding windows. :ghpull:`199`

- **New Feature**

    - Color encoding matched sources in the data source tree. :ghpull:`220`
    - Add processed pulse counter in ImageTool. :ghpull:`216`
    - Add support for C++ API. :ghpull:`213`
    - Add xtensor-blas as a submodule. :ghpull:`210`
    - Implement image transform processor and view (FFT, edge detection). :ghpull:`203`
    - Integrate Karabo gate (PipeToEXtraFoam device) which allows request pipeline
      and control data in special suite. :ghpull:`168`


0.8.3 (11 May 2020)
------------------------

- **Breaking change**
    - In the terminal, "--n_modules 2" is required to run JungFrauPR with two modules. :ghpull:`41`

- **Bug Fix**
    - Change pixel size of ePix100 from 0.11 mm to 0.05 mm. :ghpull:`189`

- **Improvement**
    - Mask tile/ASIC edges by default. :ghpull:`192`
    - Improve geometry 1M and its unittest. :ghpull:`190`
    - Invert y axis for displayed image. :ghpull:`187`
    - Rename geometry to geometry_1m in C++. :ghpull:`186`
    - Improve tr-XAS analysis in special suite. :ghpull:`163`, :ghpull:`183`
    - Improve correlating error message. :ghpull:`182`
    - Improve documentation for special suite. :ghpull:`177`
    - New reset interface in special suite. :ghpull:`170`
    - Regularize names of methods and attributes in special suite. :ghpull:`167`
    - Add new mode, start/end train ID control and progress bar, etc. in FileStreamer. :ghpull:`166`
    - Move definition of meta source from config to SourceCatalog. :ghpull:`165`
    - Use correlated queue in special suite. :ghpull:`164`
    - Improve shape comparing error message in C++. :ghpull:`160`
    - Improve mask image data implementation and interface. :ghpull:`157`
    - Move image assembler into image processor. :ghpull:`155`
    - Refactor masking code. :ghpull:`149`
    - Implement generic binding for nansum and nanmean. :ghpull:`114`

- **New Feature**
    - Add axis calibration in Gotthard analysis. :ghpull:`179`
    - Implement generalized geometry for multi-module detectors. :ghpull:`175`, :ghpull:`196`
    - Implement streaming JungFrauPR data from files. :ghpull:`174`
    - Implement Gotthard pump-probe analysis in special suite. :ghpull:`173`, :ghpull:`178`
    - Add ROI histogram in CameraView in special suite. :ghpull:`172`
    - Add ROI control in special suite. :ghpull:`171`
    - Implement XAS-TIM-XMCD in special suite. :ghpull:`162`
    - Implement MultiCameraView in special suite. :ghpull:`147`
    - Implement XAS-TIM in special suite. :ghpull:`146`
    - Implement load and save mask in pixel coordinates. :ghpull:`132`, :ghpull:`154`, :ghpull:`185`, :ghpull:`191`, :ghpull:`197`


0.8.2 (8 April 2020)
------------------------

- **Bug Fix**

    - Fix not able to close file stream process when closing, if the file stream window
      is opened through the main GUI. :ghpull:`122`
    - Fix offset correction switch between dark and offset. :ghpull:`141`

- **Improvement**

    - Move mouse hover (x, y, v) display implementation to ImageViewF. :ghpull:`148`
    - Visualize dark and offset separately. :ghpull:`141`
    - Improve loading reference image and calibration constants. :ghpull:`141`
    - Implement smart auto levels of image. :ghpull:`138`
    - Enhance SourceCatalog.add_item. :ghpull:`137`
    - Improve class init with moving average descriptor. :ghpull:`136`
    - Bump EXtra-data version and remove duplicated code. :ghpull:`131`
    - Tweak assembling code in C++ to make the result exactly the same as EXtra-geom. :ghpull:`129`
    - Simplify ImageProc binding code. :ghpull:`125`
    - Update dependencies. :ghpull:`118`
    - Update documentation. :ghpull:`115`, :ghpull:`130`
    - Move tr-XAS analysis to special suite. :ghpull:`89`

- **New Feature**

    - Generalize file stream. :ghpull:`122`
    - Add standard deviation, variance and speckle contrast into ROI FOM. :ghpull:`119`
    - Implement tile edge mask for modular detectors. :ghpull:`110`
    - Add support for fast ADC as a digitizer source. :ghpull:`101`
    - Implement Camera view (special suite). :ghpull:`89`
    - Implement Gotthard analysis (special suite) for MID. :ghpull:`89`
    - Implement interface and examples for special analysis suite. :ghpull:`89`


0.8.1 (16 March 2020)
------------------------

- **Improvement**

    - Automatically reset empty image mask with inconsistent shape. :ghpull:`104`

- **New Feature**

    - Implement AGIPD 1M geometry in C++. :ghpull:`102`
    - Add ROI1_DIV_ROI2 as an option for ROI FOM. :ghpull:`103`
    - Implement normalization for ROI FOM. :ghpull:`96`
    - Implement ROI FOM master-slave scan. :ghpull:`93`
    - Add branch-based CI and Singularity image deployment. :ghpull:`92`
    - Add support for ePix100 detector. :ghpull:`90`
    - Implement save and load metadata. :ghpull:`87`


0.8.0.1 (3 March 2020)
------------------------

- **Bug Fix**

    - Fix display bug in ImageTool :ghpull:`85`


0.8.0 (2 March 2020)
------------------------

- **Improvement**

    - Get rid of the artifact induced by masking pixel to zero when calculating
      statistics, e.g. mean, median, std.
    - Provide a mask to pyFAI to perform azimuthal integration. :ghpull:`61`
    - New C++ implementation to mask pixel in Nan and/or return a boolean mask. :ghpull:`61`
    - ROI pulse FOM and NORM will only be calculated after registration. :ghpull:`61`

- **New Feature**

    - Enable train-resolved FOM filter. :ghpull:`78`
    - Display numbers of processed and dropped trains. :ghpull:`77`
    - Support online single module data from a modular detector. :ghpull:`72`
    - Allow type selection for 1D projection (sum or mean). :ghpull:`71`
    - Implement mouse cursor value indicator for PlotWidgetF. :ghpull:`66`
    - Preliminary implementation of nanmean and nansum in C++. :ghpull:`61`

- **Bug Fix**

    - Fix pulse-filter in digitizer. :ghpull:`80`
    - Fix gain/offset slicer for train-resolved detectors. :ghpull:`76`
    - Use nansum in Tr-XAS analysis. :ghpull:`75`
    - Fix typo in unittest. :ghpull:`74`
    - Fix changing device ID in data source on the fly. :ghpull:`69`


0.7.3 (24 February 2020)
------------------------

- **Breaking change**

    - In the terminal, "--topic" becomes a positional argument. :ghpull:`41`

- **Improvement**

    - Reimplement Color classes. mkPen and mkBrush from pyqtgraph are not needed
      anymore. :ghpull:`53`
    - Allow select pipeline policy (wait or drop) via commandline. The default is wait
      since the data arrival speed is slower than the processing speed during online
      analysis. :ghpull:`45`
    - Replace Python's build-in queue.Queue to speed up data transfer. :ghpull:`45`
    - Improve the visualization of heatmap. :ghpull:`44`
    - Allow starting instances with different detectors without warning message. :ghpull:`41`
    - Allow to shutdown others' Redis server to avoid zombie Redis server occupying
      the port. :ghpull:`41`
    - Implement Fast assembling for LPD and DSSC in C++. :ghpull:`40`
    - Resign the config code. Now each instrument will has its own config file,
      e.g. scs.config.yaml, fxe.config.yaml. All the instrument sources will be
      set up in the config file. :ghpull:`38`
    - Implement streaming raw (AGIPD, LPD) data from files and also 'confirmed'
      streaming raw (AGIPD, LPD) data online. :ghpull:`38`

- **New Feature**

    - Allow specific bin range of histogram. :ghpull:`56`
    - Provide ROI histogram for train-resolved detectors; Provide ROI histogram for
      the averaged image of pulse-resolved detectors. :ghpull:`56`
    - Display `mean`, `median` and `std` for all histogram plots. :ghpull:`56`
    - ROI histogram for pulse-resolved detectors. :ghpull:`55`
    - Double-y plot for 1D binning. :ghpull:`53`
    - Support normalizing by digitizer (TIM). :ghpull:`52`
    - Support multiple ZMQ endpoints connections. :ghpull:`45`
    - Automatically correlate data from the same/different endpoints with train ID. :ghpull:`45`
    - Allow automatically choosing bin range. :ghpull:`44`
    - Also add an option to stack the detectors (LPD and DSSC) without assembling. :ghpull:`40`
    - Control required sources in the DataSourceTree. :ghpull:`38`
    - Allow filtering by value for all non-detector data sources. :ghpull:`38`
    - Implement AdqDigitizer processor. :ghpull:`38`

- **Bug Fix**

    - Fix default AGIPD geometry. :ghpull:`62`
    - Disable pulse slicer for train-resolved detectors in DataSourceTree and gain/offset
      correction. :ghpull:`56`
    - Fix logger level. :ghpull:`41`
    - Fix extra-foam-kill. :ghpull:`41`


0.7.2 (16 January 2020)
-----------------------

- **Improvement**

    - Remove 'AZIMUTHAL_INTEG_RANGE' from configuration :ghpull:`32`
    - Remove 'process monitor' from action and make it a tab in DataSourceWidget :ghpull:`32`
    - Reduce the update frequency of plots which accumulates data, for example, correlation,
      histogram, heatmap, etc., to 1 Hz :ghpull:`31`
    - Improve Redis server configuration :ghpull:`29`
    - Allow ImageViewF.setImage(None) :ghpull:`28`
    - Provide better interface for users to call C++ code :ghpull:`25`
    - Log geometry change and remove 'AZIMUTHAL_INTEG_POINTS", "CENTER_X", "CENTER_Y" from
      configuration :ghpull:`24`
    - Rearrange C++ code and separate benchmark code from unittest :ghpull:`15`
    - Re-implement PairData -> SimplePairSequence and AccumulatedData -> OneWayAccuPairSequence :ghpull:`14`
    - Re-implement BinProcessor. Now, data history is stored and users can re-bin it at anytime :ghpull:`14`
    - Reduce MAX_QUEUE_SIZE from 5 to 2 to reduce latency :ghpull:`14`
    - Remove 'update_hist' in PumpProbeData and CorrelationData. Now GUI update is completely
      decoupled from processors :ghpull:`14`
    - Merge CorrelationWindow into StatisticsWindow. Rename the old statistics widgets to histogram
      widgets; add a new tab in the MainGUI which is dedicated for 'statistics' control :ghpull:`14`
    - Update dependencies :ghpull:`11`
    - Simplify ThreadLogger code :ghpull:`10`

- **New Feature**

    - Implement q-map visualization :ghpull:`32`
    - Implement pixel-wise gain-offset correction by loading numpy array from files :ghpull:`25`
    - New ROI analysis interface (enable different FOMs of ROI; enable pulse-resolved
      ROI normalizer; enable pulse-resolved ROI1 +/- ROI2 FOM; enable visualization of
      ROI projection and pulse-resolved ROI FOM in ImageTool) :ghpull:`12`

- **Bug Fix**

    - Fix a bug in MovingAverageScalar and MovingAverageArray. Setting a new
      value of None will reset the moving average instead of being ignored :ghpull:`14`


0.7.1 (4 December 2019)
-----------------------

This is the first release after migrating from EuXFEL gitlab to github!!!

- **Improvement**

    - Rename omissive fai to foam and change config folder from karaboFAI to EXtra-foam :ghpull:`6`

- **Test**
    - Migrate CI from EuXFEL gitlab to public github :ghpull:`1`


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
