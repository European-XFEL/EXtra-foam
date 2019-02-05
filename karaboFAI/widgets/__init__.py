from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .ai_ctrl_widget import AiCtrlWidget
from .geometry_ctrl_widget import GeometryCtrlWidget
from .analysis_ctrl_widget import AnalysisCtrlWidget
from .data_ctrl_widget import DataCtrlWidget

from .misc_widgets import (
    colorMapFactory, GuiLogger, InputDialogWithCheckBox, PenFactory
)

from .sample_degradation_widget import SampleDegradationWidget
from .image_view import ImageView, RoiImageView, SinglePulseImageView
from .plot_widget import SinglePulseAiWidget, MultiPulseAiWidget
from .bulletin_widget import BulletinWidget
from .correlation_ctrl_widget import CorrelationCtrlWidget
from .pump_probe_ctrl_widget import PumpProbeCtrlWidget


__all__ = [
    "colorMapFactory",
    "BulletinWidget",
    "ImageView",
    "MultiPulseAiWidget",
    "RoiImageView",
    "SampleDegradationWidget",
    "SinglePulseAiWidget",
    "SinglePulseImageView",
]

# add control widgets
__all__.extend([
    "AiCtrlWidget",
    "AnalysisCtrlWidget",
    "CorrelationCtrlWidget",
    "DataCtrlWidget",
    "GeometryCtrlWidget",
    "PumpProbeCtrlWidget"
])

# miscellaneous
__all__.extend([
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "GuiLogger",
    "InputDialogWithCheckBox",
    "PenFactory"
])