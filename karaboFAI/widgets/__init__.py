from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .misc_widgets import (
    CustomGroupBox, FixedWidthLineEdit, GuiLogger, InputDialogWithCheckBox
)
from .plot_widgets import MainGuiImageViewWidget, MainGuiLinePlotWidget


__all__ = [
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "GuiLogger",
    "InputDialogWithCheckBox",
    "MainGuiLinePlotWidget",
    "MainGuiImageViewWidget"
]
