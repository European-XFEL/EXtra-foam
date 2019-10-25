from .image_views import (
    Bin1dHeatmap, Bin2dHeatmap, ImageAnalysis,
    ImageViewF, PumpProbeImageView, RoiImageView, SinglePulseImageView
)

# add image views
__all__ = ([
    "Bin1dHeatmap",
    "Bin2dHeatmap",
    "ImageAnalysis",
    "ImageViewF",
    "PumpProbeImageView",
    "RoiImageView",
    "SinglePulseImageView",
])

from .plot_widgets import (
    Bin1dHist,
    CorrelationWidget,
    FomHistogramWidget,
    PoiStatisticsWidget,
    PulsesInTrainFomWidget,
    PumpProbeOnOffWidget,
    PumpProbeFomWidget,
    TrainAiWidget,
)

# add plot widgets
__all__.extend([
    "Bin1dHist",
    "CorrelationWidget",
    "FomHistogramWidget",
    "PoiStatisticsWidget",
    "PumpProbeOnOffWidget",
    "PumpProbeFomWidget",
    "PumpProbeOnOffWidget",
    "TrainAiWidget",
])
