from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .misc_widgets import (
    colorMapFactory, CustomGroupBox, FixedWidthLineEdit, GuiLogger,
    InputDialogWithCheckBox
)

from .sample_degradation_widget import SampleDegradationWidget
from .image_analysis_widget import ImageAnalysisWidget, SinglePulseImageWidget
from .single_pulse_ai_widget import SinglePulseAiWidget
from .multi_pulse_ai_widget import MultiPulseAiWidget
from .bulletin_widget import BulletinWidget


__all__ = [
    "colorMapFactory",
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "GuiLogger",
    "InputDialogWithCheckBox",
    "BulletinWidget",
    "ImageAnalysisWidget",
    "MultiPulseAiWidget",
    "SampleDegradationWidget",
    "SinglePulseAiWidget",
    "SinglePulseImageWidget"
]
