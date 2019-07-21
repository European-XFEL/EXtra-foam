from .base_processor import (
    _BaseProcessor, SharedProperty, StopCompositionProcessing
)
from .azimuthal_integration import (
    AzimuthalIntegrationProcessorTrain,
    AzimuthalIntegrationProcessorPulse,
)
from .bin import BinProcessor
from .correlation import CorrelationProcessor
from .image_processor import ImageProcessorPulse, ImageProcessorTrain
from .image_assembler import ImageAssemblerFactory
from .pump_probe_processor import PumpProbeProcessor
from .roi import RoiProcessor
from .xas import XasProcessor
from .xgm import XgmProcessor
from .pulses_in_train import PulsesInTrainProcessor
