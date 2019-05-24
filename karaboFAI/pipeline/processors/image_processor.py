"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import LeafProcessor, CompositeProcessor, SharedProperty
from ..exceptions import ProcessingError
from ...metadata import Metadata as mt
from ...helpers import profiler
from ..data_model import ProcessedData


class ImageProcessor(CompositeProcessor):
    """ImageProcessor.

    Attributes:
    """
    background = SharedProperty()
    threshold_mask = SharedProperty()
    ma_window = SharedProperty()

    def __init__(self):
        super().__init__()

    def update(self):
        cfg = self._meta.get_all(mt.IMAGE_PROC)
        if cfg is None:
            return

        self.background = float(cfg['background'])
        self.threshold_mask = self.str2tuple(cfg['threshold_mask'],
                                             handler=float)
        self.ma_window = int(cfg['ma_window'])

    def process_image(self, tid, assembled):
        processed = ProcessedData(tid, assembled)

        processed.image.set_ma_window(self.ma_window)
        processed.image.set_background(self.background)
        processed.image.set_threshold_mask(*self.threshold_mask)

        return processed
