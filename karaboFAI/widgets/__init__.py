from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .ai_ctrl_widget import AiCtrlWidget
from .geometry_ctrl_widget import GeometryCtrlWidget
from .analysis_ctrl_widget import AnalysisCtrlWidget
from .data_ctrl_widget import DataCtrlWidget

from .misc_widgets import (
    colorMapFactory, GuiLogger, InputDialogWithCheckBox, make_brush, make_pen
)

from .sample_degradation_widget import SampleDegradationWidget
from .image_view import ImageView, RoiImageView, SinglePulseImageView
from .plot_widget import (
    CorrelationWidget, LaserOnOffAiWidget, LaserOnOffFomWidget,
    MultiPulseAiWidget, SinglePulseAiWidget, RoiValueMonitor
)
from .bulletin_widget import BulletinWidget
from .correlation_ctrl_widget import CorrelationCtrlWidget
from .pump_probe_ctrl_widget import PumpProbeCtrlWidget


# miscellaneous
__all__ = [
    "BulletinWidget",
    "colorMapFactory",
    "GuiLogger",
    "InputDialogWithCheckBox",
    "make_brush",
    "make_pen"
]

# add plot widgets
__all__.extend([
    "CorrelationWidget",
    "LaserOnOffAiWidget",
    "LaserOnOffFomWidget",
    "MultiPulseAiWidget",
    "RoiValueMonitor",
    "SampleDegradationWidget",
    "SinglePulseAiWidget",
])

# add image widgets
__all__.extend([
    "ImageView",
    "RoiImageView",
    "SinglePulseImageView",
])

# add control widgets
__all__.extend([
    "AiCtrlWidget",
    "AnalysisCtrlWidget",
    "CorrelationCtrlWidget",
    "DataCtrlWidget",
    "GeometryCtrlWidget",
    "PumpProbeCtrlWidget"
])
