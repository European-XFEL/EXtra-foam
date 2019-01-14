from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .ai_ctrl_widget import AiCtrlWidget
from .geometry_ctrl_widget import GeometryCtrlWidget
from .analysis_ctrl_widget import AnalysisCtrlWidget
from .data_ctrl_widget import DataCtrlWidget

from .misc_widgets import (
    colorMapFactory, GuiLogger, InputDialogWithCheckBox
)

from .sample_degradation_widget import SampleDegradationWidget
from .image_analysis_widget import ImageAnalysisWidget, SinglePulseImageWidget
from .single_pulse_ai_widget import SinglePulseAiWidget
from .multi_pulse_ai_widget import MultiPulseAiWidget
from .bulletin_widget import BulletinWidget


__all__ = [
    "colorMapFactory",
    "BulletinWidget",
    "ImageAnalysisWidget",
    "MultiPulseAiWidget",
    "SampleDegradationWidget",
    "SinglePulseAiWidget",
    "SinglePulseImageWidget",
]

# add control widgets
__all__.extend([
    "AiCtrlWidget",
    "AnalysisCtrlWidget",
    "DataCtrlWidget",
    "GeometryCtrlWidget",
])

# miscellaneous
__all__.extend([
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "GuiLogger",
    "InputDialogWithCheckBox",
])