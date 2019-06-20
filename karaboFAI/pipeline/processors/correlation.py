"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

CorrelationProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import (
    LeafProcessor, CompositeProcessor, _get_slow_data, SharedProperty
)
from ..exceptions import ProcessingError
from ...algorithms import slice_curve
from ...config import config, CorrelationFom
from ...metadata import Metadata as mt
from ...utils import profiler


class CorrelationProcessor(CompositeProcessor):
    """Add correlation information into processed data.

    Attributes:
        fom_type (CorrelationFom): type of the figure-of-merit
        fom_integ_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    fom_integ_range = SharedProperty()
    fom_type = SharedProperty()
    device_ids = SharedProperty()
    properties = SharedProperty()

    def __init__(self):
        super().__init__()

        n_params = len(config["CORRELATION_COLORS"])
        self.device_ids = [None] * n_params
        self.properties = [None] * n_params

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.CORRELATION_PROC)
        if cfg is None:
            return

        self.fom_type = CorrelationFom(int(cfg['fom_type']))

        self.fom_integ_range = self.str2tuple(
            self._meta.get(mt.AZIMUTHAL_INTEG_PROC, 'integ_range'))

        for i in range(len(self.device_ids)):
            self.device_ids[i] = cfg[f'device_id{i+1}']
            self.properties[i] = cfg[f'property{i+1}']

    @profiler("Correlation Processor")
    def process(self, data):
        """Override."""
        if self.fom_type is None or self.fom_type == CorrelationFom.UNDEFINED:
            return

        processed = data['processed']

        if self.fom_type == CorrelationFom.PUMP_PROBE:
            fom = processed.pp.fom
            # Don't raise an Exception here if fom is None since it does not
            # work well if on- and off- pulses are in different trains.

        elif self.fom_type == CorrelationFom.ROI1:
            fom = processed.roi.roi1_fom
            if fom is None:
                raise ProcessingError("[Correlation] ROI1 is not activated!")

        elif self.fom_type == CorrelationFom.ROI2:
            fom = processed.roi.roi2_fom
            if fom is None:
                raise ProcessingError("[Correlation] ROI2 is not activated!")

        elif self.fom_type in (CorrelationFom.ROI_SUM, CorrelationFom.ROI_SUB):
            fom1 = processed.roi.roi1_fom
            if fom1 is None:
                raise ProcessingError("[Correlation] ROI1 is not activated!")
            fom2 = processed.roi.roi2_fom
            if fom2 is None:
                raise ProcessingError("[Correlation] ROI2 is not activated!")

            if self.fom_type == CorrelationFom.ROI_SUM:
                fom = fom1 + fom2
            else:
                fom = fom1 - fom2

        elif self.fom_type == CorrelationFom.AZIMUTHAL_INTEG_MEAN:
            momentum = processed.ai.momentum
            if momentum is None:
                raise ProcessingError(
                    "[Correlation] Azimuthal integration result is not "
                    "available!")
            intensity = processed.ai.intensity_mean

            # calculate figure-of-merit
            fom = slice_curve(intensity, momentum, *self.fom_integ_range)[0]
            fom = np.sum(np.abs(fom))

        else:
            name = str(self.fom_type).split(".")[-1]
            raise ProcessingError(f"[Correlation] Unknown FOM name: {name}!")

        if fom is None:
            return

        # set the FOM and correlator values
        processed.correlation.fom = fom
        for i, (dev_id, ppt) in enumerate(zip(self.device_ids,
                                              self.properties)):
            if not dev_id or not ppt:
                continue

            try:
                ret = _get_slow_data(processed.tid, data['raw'], dev_id, ppt)
            except ProcessingError as e:
                raise ProcessingError(f"[Correlation] {str(e)}")

            setattr(processed.correlation, f'correlator{i+1}', ret)
