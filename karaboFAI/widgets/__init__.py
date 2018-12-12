from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .misc_widgets import (
    CustomGroupBox, FixedWidthLineEdit, GuiLogger, InputDialogWithCheckBox
)


__all__ = [
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "GuiLogger",
    "InputDialogWithCheckBox"
]
