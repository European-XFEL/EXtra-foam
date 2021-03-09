# Architecture
This document gives a high-level overview of EXtra-foam's codebase.

There are two languages used: C++ and Python. C++ is used to implement a number
of analysis algorithms, e.g. azimuthal integration. Everything else (GUI, etc)
is implemented in Python, using the Python bindings for the C++ code (which is
done with [Pybind11](https://pybind11.readthedocs.io/en/stable/)). Redis is used
for message passing.

## Code map
Here we describe the code in the different directories.

### `src/extra_foam`
This holds the implementations of various analysis algorithms in C++, and their
Python bindings.

### `extra_foam/configs`
Configuration files for each instrument group.

### `extra_foam/algorithms`
Tests and helper functions that make use of the algorithms implemented in C++.

### `extra_foam/gui`
Contains all of the code for the main GUI.

The `main GUI` is a `QMainWindow` that serves as the aggregator of the GUI
elements and the pipeline components for receiving data. It also has signals
that provides control on the running state of the process workers and the
image and plot windows. This can be found on the toolbar buttons.

The toolbar also gives entry point for the following windows:

- `ImageToolWindow`, which is a second main GUI for image manipulation (e.g.,
  selecting ROI, masking, normalization).

- Plot windows (which inherits from `_AbstractPlotWindow`):
  - `PulseOfInterestWindow`
  - `PumpProbeWindow`
  - `CorrelationWindow`
  - `HistogramWindow`
  - `BinningWindow`

- Satellite windows (which inherits from `_AbstractSatelliteWindow`). These are
auxiliary windows for additional functionalities and display:
  - `FileStreamWindow`
  - `AboutWindow`

On its interface, one can change the run configuration via the control widgets
(which inherits from `_AbstractCtrlWidget`.). These are for:

- data source management (`DataSourceWidget`)
- extensions (`ExtensionCtrlWidget`)
- general analysis
  - `AnalysisCtrlWidget`
  - `FomFilterCtrlWidget`
- utilities
  - logger
  - analysis setup manager

### `extra_foam/special_suite`
The code for all the special suites live here. There are pairs of files that
hold the implementations for each app in the suite, e.g. `foobar_w.py` would
contain GUI code and `foobar_proc.py` the code for any processing.

The main window for each app should inherit from the `_SpecialAnalysisBase`
class. Three other classes are required:
- A control widget class which inherits from `_BaseAnalysisCtrlWidgetS`, this
  holds any necessary extra UI widgets.
- A processing class which inherits from `QThreadWorker`, this handles all of
  the processing required.
- A client class that receives the data from some server to pass it to the
  processing class. The `QThreadKbClient`, which uses the
  [karabo_bridge](https://github.com/European-XFEL/karabo-bridge-py/) client,
  should be suitable in most cases.
