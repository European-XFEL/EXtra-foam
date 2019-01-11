from .pyqtgraph import setConfigOption
setConfigOption("imageAxisOrder", "row-major")

from .control_widgets import (
    AiSetUpWidget, DataSrcWidget, AnalysisSetUpWidget, GmtSetUpWidget
)
from .misc_widgets import (
    colorMapFactory, FixedWidthLineEdit, GuiLogger, InputDialogWithCheckBox
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
    "AiSetUpWidget",
    "AnalysisSetUpWidget",
    "DataSrcWidget",
    "FileServerWIdget",
    "GmtSetUpWidget",
])

# miscellaneous
__all__.extend([
    "CustomGroupBox",
    "FixedWidthLineEdit",
    "GuiLogger",
    "InputDialogWithCheckBox",
])