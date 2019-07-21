from .azimuthal_integ_ctrl_widget import AzimuthalIntegCtrlWidget
from .analysis_ctrl_widget import AnalysisCtrlWidget
from .bin_ctrl_widget import BinCtrlWidget
from .correlation_ctrl_widget import CorrelationCtrlWidget
from .data_ctrl_widget import DataCtrlWidget
from .geometry_ctrl_widget import GeometryCtrlWidget
from .pump_probe_ctrl_widget import PumpProbeCtrlWidget
from .xas_ctrl_widget import XasCtrlWidget
from .roi_ctrl_widget import RoiCtrlWidget
from .pulses_in_train_ctrl_widget import PulsesInTrainCtrlWidget


# add control widgets
__all__ = [
    "AzimuthalIntegCtrlWidget",
    "AnalysisCtrlWidget",
    "BinCtrlWidget",
    "CorrelationCtrlWidget",
    "DataCtrlWidget",
    "PulsesInTrainCtrlWidget",
    "GeometryCtrlWidget",
    "PumpProbeCtrlWidget",
    "RoiCtrlWidget",
    "XasCtrlWidget",
]
