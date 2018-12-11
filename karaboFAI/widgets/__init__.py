from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .misc_widgets import (
    CustomGroupBox, FixedWidthLineEdit, GuiLogger, InputDialogWithCheckBox
)
from .plot_widgets import AiImageViewWidget, AiMultiLinePlotWidget


__all__ = [
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "GuiLogger",
    "InputDialogWithCheckBox",
    "AiMultiLinePlotWidget",
    "AiImageViewWidget"
]
