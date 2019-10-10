from .azimuthal_integ_ctrl_widget import AzimuthalIntegCtrlWidget
from .analysis_ctrl_widget import AnalysisCtrlWidget
from .bin_ctrl_widget import BinCtrlWidget
from .correlation_ctrl_widget import CorrelationCtrlWidget
from .data_ctrl_widget import DataCtrlWidget
from .geometry_ctrl_widget import GeometryCtrlWidget
from .pump_probe_ctrl_widget import PumpProbeCtrlWidget
from .roi_ctrl_widget import RoiCtrlWidget
from .statistics_ctrl_widget import StatisticsCtrlWidget
from .data_reduction_ctrl_widget import DataReductionCtrlWidget
from .data_source_widget import DataSourceWidget


# add control widgets
__all__ = [
    "AzimuthalIntegCtrlWidget",
    "AnalysisCtrlWidget",
    "BinCtrlWidget",
    "CorrelationCtrlWidget",
    "DataCtrlWidget",
    "DataReductionCtrlWidget",
    "DataSourceWidget",
    "StatisticsCtrlWidget",
    "GeometryCtrlWidget",
    "PumpProbeCtrlWidget",
    "RoiCtrlWidget",
]
