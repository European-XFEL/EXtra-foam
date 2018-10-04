from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .misc_widgets import (
    CustomGroupBox, FixedWidthLineEdit, InputDialogWithCheckBox
)
from .plot_widgets import MainGuiImageViewWidget, MainGuiLinePlotWidget
from .plot_windows import (
    BraggSpotsWindow, DrawMaskWindow, IndividualPulseWindow, LaserOnOffWindow,
    SampleDegradationMonitor
)


__all__ = [
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "InputDialogWithCheckBox",
    "MainGuiLinePlotWidget",
    "MainGuiImageViewWidget",
    "BraggSpotsWindow",
    "DrawMaskWindow",
    "IndividualPulseWindow",
    "LaserOnOffWindow",
    "SampleDegradationMonitor"
]
