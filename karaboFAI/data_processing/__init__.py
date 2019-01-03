from .data_model import DataSource, ProcessedData
from .proc_utils import (
    down_sample, nanmean_axis0_para, normalize_curve, slice_curve, up_sample
)
from .comanalysis import COMDataProcessor
from .fai import DataProcessor
# ProcessedData class is general so we can use it for COM Analysis too.
# To separate data_processing.py for different entry points
# TODO: There could be a better way for specific import depending on
#       entry point
# import pkg_resources
# for entrypoint in pkg_resources.iter_entry_points("console_scripts"):
#     if entrypoint.name == "bragg-gui":
#         from .comanalysis.data_processing_new import DataProcessor
#     else:
#         from .fai.data_processing import DataProcessor

__all__ = [
    'DataSource',
    'ProcessedData',
    'down_sample',
    'nanmean_axis0_para',
    'normalize_curve',
    'slice_curve',
    'up_sample',
    'COMDataProcessor',
    'DataProcessor'
]
