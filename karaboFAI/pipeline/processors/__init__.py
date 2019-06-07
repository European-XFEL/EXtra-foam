from .base_processor import (
    _BaseProcessor, SharedProperty, StopCompositionProcessing
)
from .azimuthal_integration import AzimuthalIntegrationProcessor
from .correlation import CorrelationProcessor
from .pump_probe import PumpProbeProcessor
from .roi import RoiProcessor
from .xas import XasProcessor
from .bin import BinProcessor
from .image_processor import ImageProcessor
from .image_assembler import ImageAssemblerFactory
from .xgm import XgmProcessor
