"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

XgmProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError
from ...database import Metadata as mt
from ...utils import profiler


class XgmProcessor(_BaseProcessor):
    """Process XGM data."""

    def __init__(self):
        super().__init__()

        self._xgm_src = None

    def update(self):
        cfg = self._meta.get_all(mt.DATA_SOURCE)

        self._xgm_src = cfg['xgm_source_name']

    @profiler("XGM Processor")
    def process(self, data):
        """Processor XGM train data."""
        xgm_src = self._xgm_src

        processed = data['processed']
        raw = data['raw']

        if not xgm_src:
            processed.xgm.source = None
            processed.xgm.intensity = 0
            return

        if xgm_src not in raw:
            raise ProcessingError(
                f"XGM device '{xgm_src}' is not in the data!")

        xgm_data = raw[xgm_src]
        ppt = 'pulseEnergy.photonFlux'
        if ppt not in xgm_data:
            ppt = f"{ppt}.value"  # From the file

        processed.xgm.source = xgm_src
        processed.xgm.intensity = xgm_data[ppt]
