from .azimuthal_integ_ctrl_widget import AzimuthalIntegCtrlWidget
from .analysis_ctrl_widget import AnalysisCtrlWidget
from .bin_ctrl_widget import BinCtrlWidget
from .correlation_ctrl_widget import CorrelationCtrlWidget
from .geometry_ctrl_widget import GeometryCtrlWidget
from .pump_probe_ctrl_widget import PumpProbeCtrlWidget
from .projection1d_ctrl_widget import Projection1DCtrlWidget
from .statistics_ctrl_widget import StatisticsCtrlWidget
from .pulse_filter_ctrl_widget import PulseFilterCtrlWidget
from .data_source_widget import DataSourceWidget


# add control widgets
__all__ = [
    "AzimuthalIntegCtrlWidget",
    "AnalysisCtrlWidget",
    "BinCtrlWidget",
    "CorrelationCtrlWidget",
    "PulseFilterCtrlWidget",
    "DataSourceWidget",
    "StatisticsCtrlWidget",
    "GeometryCtrlWidget",
    "PumpProbeCtrlWidget",
    "Projection1DCtrlWidget",
]
